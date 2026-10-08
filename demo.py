import streamlit as st
import joblib
import pandas as pd

# 加载模型
model_prod = joblib.load("model_prod.pkl")
model_inc = joblib.load("model_inc.pkl")

st.title("农产品“产量‑收益”双目标决策系统")
st.subheader("面向小农户的轻量化助农决策方案")

year = st.number_input("预测年份", min_value=1960, max_value=2040, value=2026)
area = st.number_input("种植面积（公顷）", min_value=1000, value=10000)
yield_val = st.number_input("单产（kg/ha）", min_value=1000, value=5000)

input_df = pd.DataFrame({
    "Year":[year],
    "Area harvested":[area],
    "Yield":[yield_val]
})

if st.button("开始预测"):
    pred_prod = model_prod.predict(input_df)[0]
    pred_inc = model_inc.predict(input_df)[0]
    st.success("✅预测完成")
    st.write(f"预估总产量：{pred_prod:.2f} 吨")
    st.write(f"预估净收益：{pred_inc:.2f} 元")
    st.info("📌农事建议：扩大种植面积可以提高总产量，但农资、土地成本同步增加，净收益不一定持续上升，可多次调整参数权衡。")

st.markdown("本系统基于集成学习+SHAP可解释AI，打破算法黑盒，输出农户看得懂的农事建议。")