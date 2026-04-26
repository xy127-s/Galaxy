import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 设置页面配置
st.set_page_config(
    page_title="Streamlit 演示应用",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面标题
st.title("Streamlit 功能演示")
st.markdown("---")

# 侧边栏
st.sidebar.title("控制面板")

# 数据可视化部分
st.subheader("📈 数据可视化")

# 生成随机数据
np.random.seed(42)
data = pd.DataFrame({
    'x': np.arange(100),
    'y1': np.random.normal(0, 1, 100).cumsum(),
    'y2': np.random.normal(0, 2, 100).cumsum(),
    'y3': np.random.normal(0, 0.5, 100).cumsum()
})

# 交互式控件
chart_type = st.selectbox("选择图表类型", ["折线图", "散点图", "面积图"])
show_grid = st.checkbox("显示网格", value=True)

# 绘制图表
fig, ax = plt.subplots(figsize=(12, 6))

if chart_type == "折线图":
    ax.plot(data['x'], data['y1'], label='数据1')
    ax.plot(data['x'], data['y2'], label='数据2')
    ax.plot(data['x'], data['y3'], label='数据3')
elif chart_type == "散点图":
    ax.scatter(data['x'], data['y1'], label='数据1')
    ax.scatter(data['x'], data['y2'], label='数据2')
    ax.scatter(data['x'], data['y3'], label='数据3')
elif chart_type == "面积图":
    ax.fill_between(data['x'], data['y1'], alpha=0.5, label='数据1')
    ax.fill_between(data['x'], data['y2'], alpha=0.5, label='数据2')
    ax.fill_between(data['x'], data['y3'], alpha=0.5, label='数据3')

ax.set_title(f"{chart_type} 演示")
ax.set_xlabel("X 轴")
ax.set_ylabel("Y 轴")
if show_grid:
    ax.grid(True)
ax.legend()

st.pyplot(fig)

st.markdown("---")

# 文本处理部分
st.subheader("📝 文本处理")

user_input = st.text_area("输入文本", "这是一个文本处理演示。\n你可以输入任何文本，系统会统计字符数、单词数和行数。")

if user_input:
    char_count = len(user_input)
    word_count = len(user_input.split())
    line_count = len(user_input.split('\n'))
    
    st.write(f"**字符数**: {char_count}")
    st.write(f"**单词数**: {word_count}")
    st.write(f"**行数**: {line_count}")
    
    # 文本转换
    st.subheader("文本转换")
    transform_option = st.radio(
        "选择转换方式",
        ["转为大写", "转为小写", "首字母大写"]
    )
    
    if transform_option == "转为大写":
        transformed_text = user_input.upper()
    elif transform_option == "转为小写":
        transformed_text = user_input.lower()
    else:
        transformed_text = user_input.title()
    
    st.write("转换结果:")
    st.code(transformed_text)

st.markdown("---")

# 机器学习演示部分
st.subheader("🤖 机器学习演示")

# 加载 iris 数据集
iris = load_iris()
X, y = iris.data, iris.target

# 分割数据集
train_size = st.slider("训练集比例", 0.1, 0.9, 0.7, 0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)

# 训练模型
n_estimators = st.number_input("决策树数量", 10, 200, 100, 10)
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"**模型准确率**: {accuracy:.4f}")

# 显示混淆矩阵
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
ax.set_title("混淆矩阵")
ax.set_xlabel("预测标签")
ax.set_ylabel("真实标签")

st.pyplot(fig)

# 特征重要性
feature_importance = model.feature_importances_
feature_names = iris.feature_names

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
ax.set_title("特征重要性")
ax.set_xlabel("重要性")

st.pyplot(fig)

st.markdown("---")

# 关于部分
st.subheader("ℹ️ 关于此应用")
st.write("这是一个 Streamlit 演示应用，展示了 Streamlit 的多种功能：")
st.write("- 数据可视化（折线图、散点图、面积图）")
st.write("- 交互式控件（下拉选择、复选框、滑块、单选按钮）")
st.write("- 文本处理和转换")
st.write("- 简单的机器学习模型训练和评估")
st.write("- 响应式布局设计")

st.markdown("---")
st.write("© 2026 Streamlit 演示应用")