# QA-defense-system-backend
Tongji University Spring 2024 Information Security Course Final Project：Question Answering Defense System Based on Language Model (Back-end implementation)

## 项目结构

```bash
QA-defense-system-backend
    app/
        __init__.py
        classify.py           # 视图文件
        predict_similarity.py #相似度预测
        predict_harmful.py    #有害无害预测
        mymodel.py            #有害无害模型类
        FintuneBertModel.py   #相似度模型类
        bert_config.py        #bert配置文件
        utils.py              #工具类文件
        checkpoints/
            train/            #这个文件夹放训练好的有害无害分类模型
        similarity_model/
            bert-base-chinese/  #从huggingface下载的bert-base-chinese模型文件
        model_hub/
        	chinese-bert-wwm   #hunggingface下载的chinese-bert-wwm模型
        similarity_epochs8_lr2e-5_batch128/  #训练好的相似度模型文件
            checkpoint
    requirements.txt  #项目所需依赖项
```



## 运行说明

> python3.8

1. 按照`项目结构`添加响应的模型文件夹：`checkpoints`、`similarity_model`、`similarity_epochs8_lr2e-5_batch128`

2. 在`QA-defense-system-backend`目录下新建虚拟环境

   ```
   python3 -m venv venv
   ```

3. 激活虚拟环境

   ```
   source venv/bin/activate
   ```

4. 添加依赖项

   ```
   pip install -r requirements.txt
   ```

5. 在`QA-defense-system-backend`目录下运行

   ```
   flask --app app run --debug 
   ```

   



