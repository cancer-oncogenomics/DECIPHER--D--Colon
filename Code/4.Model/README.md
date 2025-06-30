# gsml简介

## 一、项目背景

gsml是一个主要用于公司内部Mercury项目，机器学习方向的框架。

## 二、安装

不用安装

## 三、使用

[3.2 命令行工具](doc/doc_command.md)

### 3.2 命令行工具

#### h2o automl训练

> ./gsml  TrainH2oAutoML

使用H2o automl模块，对数据集和特征进行训练和预测，并保存相关base model。

+ > **Parameters**
    + --model_id: 模型ID, 也是输出文件的前缀。[required]
    + --d_output： 输出文件路径。 [required]
    + --feature： 特征文件论。 [required] [multiple]
    + --train_dataset: 训练集路径（info.list）。 [required] [multiple]
    + --pred_dataset: 预测数据集路径（info.list）。 [required] [multiple]
    + --leaderboard_dataset: 用于指导训练的数据集路径（info.list）。 [required] [multiple]
    + --nthreads： h2o init所用最大线程数。[default: 10]
    + --max_mem_size:  h2o init所用最大内存。[default: 20000M]
    + --max_models: automl训练允许的最大子模型数量。[default: 200]
    + --max_runtime_secs_per_model：每一个子模型训练的最大时长(秒)。[default: 1800]
    + --max_runtime_secs：整个automl过程的最大时长（秒）。[default: 0(不限制)]
    + --nfolds：单模型交叉验证分组数。[default: 5]
    + --seed：模型训练随机数种子。[default: -1]
    + --stopping_metric：迭代终止的指标。[default: aucpr] ["AUTO", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr", "lift_top_group", "misclassification", "mean_per_class_error", "r2"]
    + --sort_metric: 模型排序的指标。[default: aucpr] ["auc", "aucpr", "logloss", "mean_per_class_error", "rmse", "mse"]
    + --stopping_tolerance: Specify the relative tolerance for the metric-based stopping criterion to stop a grid search and the training of individual models within the AutoML run. [default: 0.001]

+ > **Examples**

    ```bash
    ./gsml TrainH2oAutoML \
        --model_id test \
        --d_output . \
        --feature cnv.csv \
        --train_dataset train.info.list \
        --pred_dataset Valid1.info.list \
        --pred_dataset Valid2.info.list \
        --pred_dataset Test.info.list
    ```

#### 模型性能统计

> ./gsml  ModelStat

根据模型预测结果，以及数据集和优化项目信息，统计分析模型的具体性能。可以是一个gsml模型实例或者是一个预测得分文件。

+ > **Parameters**

    + --f_model: gsml模型路径。（f_model和f_score指定一个即可）
    + --f_score：模型预测得分文件路径。（必须包含SampleID和Score）
    + --dataset：数据集路径。（格式为：数据集名称,数据集路径）
    + --optimize：优化项目文件路径。（格式为：项目名称,项目路径）
    + --conf：combine score配置信息。如果需要计算combine score则该参数是必须的。
    + --spec_list：需要统计的不同的spec下的cutoff。[multiple] [default: [0.9, 0.05, 0.98]]
    + --stat: 需要统计的内容。[default: 都统计] ["auc", "score", "pred_classify", "combine_score", "performance"]
    + --d_output: 结果输出路径

+ > **Examples**

    ```bash
    ./gsml ModelStat \
        --f_model test.gsml \
        --dataset Train,Train.info.list \
        --dataset Valid1,Valid1.info.list \
        --dataset Valid2,Valid2.info.list \
        --dataset Test,Test.info.list \
        --conf gsml.test.yaml \
        --d_output .
    ```



#### 模型预测

> ./gsml Predict

使用已有的gsml模型，预测新的数据集

+ > **Parameters**

    + --f_model: gsml模型路径。[required]
    + --feature: 待预测样本的特征文件路径。[required]
    + --dataset: 带预测样本的数据集文件路径。(若指定，则取其与feature的交集样本来预测)
    + --nthreads： 模型预测使用的最大线程数。[default: 5]
    + --max_mem_size：模型预测使用的最大内存. [default: 20000M]

## 四、模块介绍

### 4.1 gsml模块

#### gsml.connect

> gsml.connect(*url=None, ip=None, port=None, **kwargs*)

连接一个本地的h2o server

+ > **Parameters**

    + url:  Full URL of the server to connect to (can be used instead of ip + port + https).

    + ip: The ip address (or host name) of the server where H2O is running.

    + port: Port number that H2O service is listening to.

    + kwargs: 兼容h2o.connet方法的所有参数

+ > **Returns**

    None

+ > **Examples**

    ```python
    import gsml
    
    gsml.connect(url="http://10.1.2.53:1134")
    gsml.connect(ip="10.1.2.53", port="1134")
    ```

    

#### gsml.init

> gsml.init(*retry=5, **kwargs*)

尝试连接到本地服务器，如果不成功，则启动新服务器并连接。初始化一个本地h2o服务

+ > **Parameters**

    + retry:  连接重试的次数。[default: 5]

    + kwargs: 兼容h2o.init方法的所有参数

+ > **Returns**

    None

+ > **Examples**

    ```python
    import gsml
    
    gsml.init(retry=5, nthreads=5, max_mem_size="20000M")
    ```

    

#### gsml.cluster

>gsml.cluster(name, workdir, n_nodes, threads, memory, version=None, port=None)

在本地集群上开启一个多节点组成的h2o server，主要用于大数量的特征训练。

+ > **Parameters**

    + name: 待开启的h2o server的名称，该名称不能与已经存在的h2o server重名。

    + workdir: 保存h2o server运行状态的目录。

    + n_nodes: 要启动的节点数量。

    + threads: 每个节点所使用的cpu数量。[default：5]

    + memory: 每个节点所使用的内存大小。[default: 14g]

    + version: 要启动的h2o版本。 [default: "3.36.0.3"]

    + port: 要使用的端口号。若不提供，则随机选择

+ > **Returns**

    None。（服务状态见workdir目录）

+ > **Examples**

    ```python
     import gsml 
     gsml.cluster(name="h2o", workdir="./", n_nodes=3, threads=5, memory="10g", port=1134)
    ```



#### gsml.close

> gsml.close()

关闭一个h2o server

+ > **Parameters**

+ > **Returns**

+ > **Examples**



#### gsml.load_model

> gsml.load_model(f_model, use_predict=False) 

载入一个gsml模型实例

+ > **Parameters**

    + f_model: 需要载入的gsml模型的路径（模型一般以.gsml为后缀）

    + use_predict: 模型载入后是否需要用来预测新数据集。如果为True的话，会在载入gsml实例的同时再载入对应h2o或者sklearn的模型（比较消耗时间）。[default: False]

+ > **Returns**

    GsEstimators

+ > **Examples**

    ```python
    import gsml
    
    model = gsml.load_model(f_model="gsml.deeplearn.gsml", use_predict=True)
    ```

    

#### gsml.save_model

> gsml.save(model, path) 

保存一个gsml模型实例。

当一个模型在做完预测之后，如果不保存该模型的话，则本次预测结果不会保存在模型中。

+ > **Parameters**

    + model: 需要保存的一个模型实例

    + path: 模型保存路径

+ > **Returns**

    None

+ > **Examples**

    ```python
    import gsml
    from gsml.estimators.deeplearning import H2ODeepLearning
    
    gf_train = gsml.GsFrame(dataset_list=["/Train.info.list"], feature_list=["Train.cnv.csv"])
    
    model = H2ODeepLearning(model_id="gsml.deeplearn")
    model.train(x=gf_train.c_features, y="Response", training_frame=gf_train)
    gsml.save_model(model, path="/dssg/home/sheny/test/demo/model")
    ```

    

### 4.2 gsml数据格式

#### GsFrame

> class gsml.GsFrame(dataset_list=None, feature_list=None, axis=0)

gsml框架中，各个模块间使用的数据格式。将数据集和特征文件进行合并，得到一个GsFrame实例。如果分别提供了数据集和特征信息，则最终展示的数据为两者之间的交集

+ > **Parameters**

    + **dataset_list**: 数据集文件（即info.list文件）。文件必须包含SampleID列和Response列，其中Response列必须为str格式，且为一个二分类数据。

    + **feature_list**: 特征文件路径。其中若是按行合并的话，需要确保各特征之间的列名唯一。

    + **axis**: 特征合并的维度。1：按行合并，0：按列合并。[default: 0]

+ > **property**

    + > **c_features**

        返回所有的特征列名。list格式

    + > **c_dataset**

        返回所有样本信息文件中的列名

    + > **samples**

        返回所有的SampleID。list格式

    + > **as_pd**

        返回合并后的数据信息。pandas DataFrame格式

    + > **as_h2o**

        返回合并后的数据信息。h2oFrame格式

    + > append(gs_frame, axis=0)

        将两个GsFrame进行合并，得到一个新的GsFrame格式的数据集。

        + > **Parameters**

            + **gs_frame**: 需要合并的GsFrame格式数据集。
            
  + **axis**: 两个数据集合并的维度。0：按列合并；1: 按行合并。[default: 0]
            
  + > Returns
        
            **GsFrame**

+ > Examples

    ```python
    import gsml
    
    gf_train = GsFrame(
        dataset_list=["Train.info.list", "Valid1.info.list"],
        feature_list=["Train.cnv.csv", "Valid1.cnv.csv"]
    )
    
    gf_train.c_features
    gf_train.samples
    gf_train.as_pd
    gf_train.as_h2o
    ```

    

### 4.3 gsml基本类

#### GsModelStat

> class gsml.model_base.GsModelStat(f_score=None, dataset=None, optimize=None, cs_conf=None)

用于模型统计的一个基本类，其统计模块主要依赖与得分文件以及数据集分类信息。

+ > **Parameters**

    + f_score：模型的训练和预测得分文件。必须包含SampleID列和
    + dataset
    + optimize
    + cs_conf

+ > **Property**

+ > **Examples**





#### GsEstimators

### 4.4 gsml机器学习模块--H2o

#### H2ODeepLearning

#### H2OGradientBoosting

#### H2OGeneralizedLinear

#### H2oRandomForest

#### H2OStackedEnsemble

#### H2OXGBoost

### 3.5 gsml机器学习模块--sklearn

## 五、项目更新方式

## 六、项目更新日志



