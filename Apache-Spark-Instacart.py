#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from kaggle.api.kaggle_api_extended import KaggleApi
import os
import zipfile
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf,when
import time
import pandas as pd

api = KaggleApi({"username":"","key":""})
api.authenticate()

files = api.competition_download_files("Instacart-Market-Basket-Analysis")
with zipfile.ZipFile('C:/Users/lefkats/Instacart-Market-Basket-Analysis.zip', 'r') as zip_ref:
    zip_ref.extractall('C:/Users/lefkats/Desktop/instacart')
working_directory = 'C:/Users/lefkats/Desktop/instacart'
os.chdir(working_directory)
for file in os.listdir(working_directory):   
    if zipfile.is_zipfile(file): 
        with zipfile.ZipFile(file) as item: 
            item.extractall()

spark = SparkSession.builder.master("yarn")    .config("spark.sql.shuffle.partitions",70)    .config("spark.driver.memory", "26g")    .config("spark.driver.cores",24)    .config("spark.executor.instances",24)    .config("spark.executor.cores",4)    .config("spark.executor.memory","26")    .appName("my-app").getOrCreate()

aisles = spark.read.csv("aisles.csv",inferSchema =True,header=True)
departments = spark.read.csv("departments.csv",inferSchema =True,header=True)
order_products_prior = spark.read.csv("order_products__prior.csv",inferSchema =True,header=True)
order_products_train = spark.read.csv("order_products__train.csv",inferSchema =True,header=True)
orders = spark.read.csv("orders.csv",inferSchema =True,header=True)
products = spark.read.csv("products.csv",inferSchema =True,header=True)

aisles.show(5,False)
departments.show(5,False)
order_products_prior.show(5,False)
order_products_train.show(5,False)
orders.show(5,False)
products.show(5,False)

print("ailses dataframe consists of:",aisles.count(),"rows and", len(aisles.columns),"columns")
print("departments dataframe consists of:",departments.count(),"rows and", len(departments.columns),"columns")
print("order_products_prior dataframe consists of:",order_products_prior.count(),"rows and", len(order_products_prior.columns),"columns")
print("order_products_train dataframe consists of:",order_products_train.count(),"rows and", len(order_products_train.columns),"columns")
print("orders dataframe consists of:",orders.count(),"rows and", len(orders.columns),"columns")
print("products dataframe consists of:",products.count(),"rows and", len(products.columns),"columns")

#create a dataFrame with the orders and the products that have been purchased on prior orders
op = orders.join(order_products_prior,on='order_id')

#calculate the total number of placed orders per customer
user = op.groupBy('user_id').agg(F.max('order_number').alias('u_total_orders')).sort('user_id')

#calculate the mean of reordered per customer
u_reorder = op.groupBy('user_id').agg(F.mean('reordered').alias('u_reordered_ratio')).sort('user_id')

#merge u_reorder dataframe to user
user = user.join(u_reorder, on='user_id', how='left').sort('user_id')

#calculate the total number of purchases for each product (from all customers)
prd = op.groupBy('product_id').agg(F.count('order_id').alias('p_total_purchases')).sort('product_id')

#remove products with less than 40 purchases
p_reorder = op.groupBy('product_id').agg(F.count('product_id').alias('total'))
p_reorder = op.join(p_reorder,on='product_id')
p_reorder = p_reorder.filter(p_reorder['total']>40)
p_reorder = p_reorder.drop('total')

#calculate the reorder probability per product and sort them by product_id
p_reorder = p_reorder.groupBy('product_id').agg(F.mean('reordered').alias('p_reorder_ratio')).sort('product_id')

#merge p_reorder dataframe to prd
prd = prd.join(p_reorder, on='product_id', how='left').sort('product_id')

#fill missing values of prd['p_reorder_ratio'] with zero
prd = prd.fillna(0, subset=['p_reorder_ratio'])

#calculate how many times a user bought a product
uxp = op.groupBy('user_id', 'product_id').agg(F.count('order_id').alias('uxp_total_bought')).sort('user_id')

times = op.groupBy('user_id', 'product_id').agg(F.count('order_id').alias('Times_Bought_N')).sort('user_id')

#calculate total_orders of each user
total_orders = op.groupBy('user_id').agg(F.max('order_number').alias('total_orders')).sort('user_id')

#calculate the order number where the user bought a product for first time
first_order_no = op.groupBy('user_id', 'product_id').agg(F.min('order_number').alias('first_order_number')).sort('user_id')

#create span dataframe with total_orders and first_order_no
span = total_orders.join(first_order_no, on='user_id', how='right').sort('user_id')

#create a new column
span = span.withColumn('Order_Range_D', span.total_orders - span.first_order_number + 1)

#create uxp_ratio dataframe with times and span
uxp_ratio = times.join(span, on=['user_id', 'product_id'], how='left').sort('user_id')

#create a column 
uxp_ratio = uxp_ratio.withColumn('uxp_reorder_ratio',uxp_ratio.Times_Bought_N / uxp_ratio.Order_Range_D).sort('user_id')

#drop columns from uxp_ratio
uxp_ratio = uxp_ratio.drop('Times_Bought_N', 'total_orders', 'first_order_number', 'Order_Range_D')

#merge uxp_ratio to uxp
uxp = uxp.join(uxp_ratio, on=['user_id', 'product_id'], how='left').sort('user_id')

#calculate 'order_number_back' which represents the 'order_number' in reserve
temp = op.groupBy('user_id').agg(F.max('order_number').alias('new'))
op = op.join(temp,on='user_id')
op = op.withColumn('order_number_back',op.new - op.order_number +1)
op = op.drop('new').sort('user_id')

#keep only the last five orders for each customer
op5 = op[op.order_number_back <= 5]

#calculate how many times each customer bought every product in their last 5 orders
last_five = op5.groupby('user_id','product_id').agg(F.count('order_id').alias('times_last5')).sort('user_id')

#merge last_five dataframe to uxp
uxp = uxp.join(last_five, on=['user_id', 'product_id'], how='left')

#fill missing values from uxp with zero
uxp = uxp.fillna(0)

#create dataframe with uxp and user
data = uxp.join(user, on='user_id', how='left')

#merge data with prd dataframe
data = data.join(prd, on='product_id', how='left')


#keep only the future orders from all customers: train & test
orders_future = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]
orders_future = orders_future[ ['user_id', 'eval_set', 'order_id'] ]

#merge orders_future to data
data = data.join(orders_future, on='user_id', how='left')

#split data to model_data and submit_data
model_data = data[data.eval_set=='train']
submit_data = data[data.eval_set=='test']

#merge order_products to model_data
model_data = model_data.join(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )

#fill missing values of model_data['reordered'] with zero
model_data = model_data.fillna(0, subset=['reordered'])

#drop columns of model_data and submit_data
model_data = model_data.drop('eval_set', 'order_id')
submit_data = submit_data.drop('eval_set','order_id')

#change the type of column "reordered"
model_data = model_data.withColumn("reordered", model_data["reordered"].cast(DoubleType()))

#create a vector features to model_data and submit_data
vectorAssembler = VectorAssembler()    .setInputCols(["uxp_total_bought", "uxp_reorder_ratio", "times_last5", "u_total_orders", "u_reordered_ratio", "p_total_purchases", "p_reorder_ratio"]).setOutputCol("features")
model_data = vectorAssembler.transform(model_data)
submit_data = vectorAssembler.transform(submit_data)
#uxp_total_bought represents how many times each customer bought each product
#uxp_reordered_ratio represents how frequent each customer bought each product
#times_last5 represents how many time each customer bought each product in his last 5 orders
#u_total_orders represents how many orders has each customer done
#u_reorder_ratio represents how frequent each customer did a reorder
#p_reordered_ratio represents how many times has each product been reordered
#p_reorder_ratio represents how frequent has each product been reordered

#keep only necessary columns
model_data = model_data.select(['product_id','user_id','reordered','features'])
submit_data = submit_data.select(['product_id','user_id','features'])
model_data.show(5)
submit_data.show(5)
#split model_data to train and test
train, test = model_data.randomSplit([0.8, 0.2])

#cache data to memory
train.cache()
test.cache()
submit_data.cache()

#print data
train.show(5)
test.show(5)
submit_data.show(5)

def classifier(name):
    start_time=time.time()
    model = name(labelCol='reordered', featuresCol='features')
    if name == LogisticRegression:
        paramGrid = ParamGridBuilder().addGrid(model.maxIter,[10,20,30]).addGrid(model.regParam,[0,0.1]).build()
    elif name == RandomForestClassifier:
        paramGrid = ParamGridBuilder().addGrid(model.maxDepth,[5,10,15]).addGrid(model.numTrees,[20,40]).build()
    elif name == GBTClassifier:
        paramGrid = ParamGridBuilder().addGrid(model.maxDepth,[5,10,15]).addGrid(model.maxBins,[20,40]).build()
    evaluator = BinaryClassificationEvaluator(labelCol='reordered')
    crossval = CrossValidator(estimator=model,estimatorParamMaps=paramGrid,evaluator=evaluator,numFolds=3) 
    cv = crossval.fit(train)
    best_model = cv.bestModel
    results = best_model.transform(test)
    time_per_model = (time.time() - start_time) / 18
    return results, time_per_model, best_model

lr_results, lr_time, lr_best_model = classifier(LogisticRegression)
rf_results, rf_time, rf_best_model = classifier(RandomForestClassifier)
gbt_results, gbt_time, gbt_best_model = classifier(GBTClassifier)

def evaluation(results):
    TN = results.filter('prediction = 0 AND reordered = prediction').count()
    TP = results.filter('prediction = 1 AND reordered = prediction').count()
    FN = results.filter('prediction = 0 AND reordered <> prediction').count()
    FP = results.filter('prediction = 1 AND reordered <> prediction').count()
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 =  2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1

lr_accuracy, lr_precision, lr_recall, lr_f1, = evaluation(lr_results)
rf_accuracy, rf_precision, rf_recall, rf_f1, = evaluation(rf_results)
gbt_accuracy, gbt_precision, gbt_recall, gbt_f1, = evaluation(gbt_results)

metrics_table = pd.DataFrame([[lr_accuracy, lr_precision, lr_recall, lr_f1, lr_time],
                              [rf_accuracy, rf_precision, rf_recall, rf_f1, rf_time],
                              [gbt_accuracy, gbt_precision, gbt_recall, gbt_f1, gbt_time]],
                              columns=['accuracy','precision','recall','f1','time_per_model'],index=['lr','rf','gbt'])
metrics_table.style.background_gradient(cmap='Reds')

def thresholded_results(results,thres):
    slicer = VectorSlicer(inputCol="probability", outputCol="probability_one", indices=[1])
    results = slicer.transform(results)
    unlist = udf(lambda x: float(list(x)[0]), DoubleType())
    results = results.withColumn("probability_one", unlist("probability_one"))
    results = results.withColumn("prediction", when(results["probability_one"] <= thres ,0).otherwise(1))
    results.cache()
    return results

lr_results1 = thresholded_results(lr_results,0.16)
rf_results1 = thresholded_results(rf_results,0.21)
gbt_results1 = thresholded_results(gbt_results,0.21)

lr_accuracy, lr_precision, lr_recall, lr_f1 = evaluation(lr_results1)
rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluation(rf_results1)
gbt_accuracy, gbt_precision, gbt_recall, gbt_f1 = evaluation(gbt_results1)

thres_table = pd.DataFrame([[lr_accuracy, lr_precision, lr_recall, lr_f1],
                              [rf_accuracy, rf_precision, rf_recall, rf_f1],
                              [gbt_accuracy, gbt_precision, gbt_recall, gbt_f1]],
                              columns=['accuracy','precision','recall','f1'],index=['lr','rf','gbt'])
thres_table.style.background_gradient(cmap='Reds')

submit_results = gbt_best_model.transform(submit_data)
submit_results = thresholded_results(submit_results,0.21)
submit_results = submit_results.select('user_id','product_id','prediction')
submit_results.show(5)

orders_test = orders.select("user_id", "order_id").filter(orders.eval_set == "test")
final = submit_results.join(orders_test, on='user_id', how='left')
final = final.drop('user_id')
final.show(5)

final = final.toPandas()
final.head()

d = dict()
for row in final.itertuples():
    if row.prediction== 1:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)
for order in final.order_id:
    if order not in d:
        d[order] = 'None'        
sub = pd.DataFrame.from_dict(d, orient='index')
sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.head(5)

sub.to_csv('sub.csv', index=False)
api.competition_submit('sub.csv','my-app',"Instacart-Market-Basket-Analysis")

