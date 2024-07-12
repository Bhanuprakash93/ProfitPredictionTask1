# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import plotly.graph_objs as go
# import matplotlib.pyplot as plt

# # Load the saved model and scalers
# best_SVR = joblib.load('best_SVR_model.pkl')
# scalary = joblib.load('scalary.pkl')
# scalarx = joblib.load('scalarx.pkl')

# # Load the data
# Corp_W_SF = pd.read_csv("C:/Users/Bhanu prakash/OneDrive - Vijaybhoomi International School/Desktop/11_07_24_Streamlit/Corp_W_SF_data.csv")

# # Define the columns used in the model
# feature_columns_NLM = ['GMRate', 'DMCoupon', 'DMPromo', 'DMOther', 'CouponPenetration', 'PromoPenetration', 'OtherPenetration', 'EconomicIndicator', 'SF']

# # Define functions
# def fn_GRID_Data_6vars(pmin, pmax, dmMin, dmMax, step_size, GMRate, DMPromo, DMOther, PromoPenetration, OtherPenetration, EconomicIndicator, SF, bestNLM, x_col, y_col):
#     x_values = np.arange(pmin, pmax, step_size)
#     y_values = np.arange(dmMin, dmMax, step_size)
#     A = pd.DataFrame({x_col: x_values})
#     B = pd.DataFrame({y_col: y_values})
#     A['key'] = 1
#     B['key'] = 1
#     df = pd.merge(A, B).drop('key', axis=1)
#     df['GMRate'] = GMRate
#     df['DMPromo'] = DMPromo
#     df['DMOther'] = DMOther
#     df['PromoPenetration'] = PromoPenetration
#     df['OtherPenetration'] = OtherPenetration
#     df['EconomicIndicator'] = EconomicIndicator
#     df['SF'] = SF

    
#     # Ensure all feature columns are present
#     for col in feature_columns_NLM:
#         if (col not in df.columns):
#             df[col] = 0  # or some default value
    
#     df_NLM = df[feature_columns_NLM]
#     scaled_df_NLM = scalarx.transform(df_NLM)
#     dfpreds_NLM = bestNLM.predict(scaled_df_NLM)
#     dfpreds_NLM_rescaled = scalary.inverse_transform(pd.DataFrame(dfpreds_NLM))
#     df['Estimated Profit'] = dfpreds_NLM_rescaled
#     return df

# def fn_3Dplot2(df, Exp_Month, x_col, y_col):
#     fig = go.Figure(data=[go.Scatter3d(
#         x=df[x_col],
#         y=df[y_col],
#         z=df['Estimated Profit'],
#         mode='markers',
#         marker=dict(size=5, color=df['Estimated Profit'], colorscale='Viridis', opacity=0.8),
#         name='Estimated_Profit'
#     )])
#     x1 = np.round(Exp_Month[x_col].iloc[0], 2)
#     y1 = np.round(Exp_Month[y_col].iloc[0], 2)
#     z1 = np.round(Exp_Month['Avg_Profit'].iloc[0], 2)
#     tit = Exp_Month['month_of_year'].iloc[0]
#     fig.add_trace(go.Scatter3d(
#         x=[x1], y=[y1], z=[z1],
#         mode='markers',
#         marker=dict(size=10, color='red', symbol='circle'),
#         name=tit+"("+str(x1)+","+str(y1)+","+str(z1)+")"
#     ))
#     fig.update_layout(
#         title=f'3D Plot of Estimated Profit vs {x_col} and {y_col}',
#         scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title='Estimated Profit')
#     )
#     st.plotly_chart(fig)

# def optim_plot2D2(df, p, ML_MODEL, x_col, y_col):
#     df_p = df[np.isclose(df[x_col], p)]
#     plt.figure(figsize=(10, 4))
#     x = df_p[y_col]
#     y = df_p['Estimated Profit']
#     plt.plot(x, y)
#     plt.xlabel(y_col)
#     plt.ylabel('Estimated Profit')
#     plt.title(f"{y_col} vs Profit when {x_col}={p}   {ML_MODEL.__class__.__name__}")
#     st.pyplot(plt)

# # Streamlit app
# st.title('Profit Prediction Analysis')
# month = st.selectbox('Select Month-Year', Corp_W_SF['month_of_year'].unique())

# # New input fields for dynamic column selection, limited to feature_columns_NLM
# x_col = st.selectbox('Select X-axis Column', feature_columns_NLM)
# y_col = st.selectbox('Select Y-axis Column', feature_columns_NLM)

# st.header('Input Values for Grid')
# pmin = st.number_input(f'Enter minimum {x_col} value', min_value=0.0, max_value=0.5, step=0.005)
# pmax = st.number_input(f'Enter maximum {x_col} value', min_value=0.0, max_value=0.5, step=0.005)
# dmMin = st.number_input(f'Enter minimum {y_col} value', min_value=0.0, max_value=0.9, step=0.005)
# dmMax = st.number_input(f'Enter maximum {y_col} value', min_value=0.0, max_value=0.9, step=0.005)
# step_size = st.number_input('Enter step size', min_value=0.001, max_value=0.1, step=0.001, value=0.005)

# if st.button('Analyze'):
#     Exp_Month = Corp_W_SF.loc[Corp_W_SF['month_of_year'] == month, ['GMRate', 'DMCoupon', 'DMPromo', 'DMOther', 'CouponPenetration', 'PromoPenetration', 'OtherPenetration', 'EconomicIndicator', 'SF', 'Avg_Profit', 'month_of_year']]
#     Grid_df = fn_GRID_Data_6vars(pmin, pmax, dmMin, dmMax, step_size,
#                                  np.round(Exp_Month['GMRate'].iloc[0], 2),
#                                  np.round(Exp_Month['DMPromo'].iloc[0], 2),
#                                  np.round(Exp_Month['DMOther'].iloc[0], 2),
#                                  np.round(Exp_Month['PromoPenetration'].iloc[0], 2),
#                                  np.round(Exp_Month['OtherPenetration'].iloc[0], 2),
#                                  np.round(Exp_Month['EconomicIndicator'].iloc[0], 2),
#                                  np.round(Exp_Month['SF'].iloc[0], 2),
#                                  best_SVR, x_col, y_col)
    
#     st.subheader('2D Plot')
#     optim_plot2D2(Grid_df, round(Exp_Month[x_col].iloc[0], 2), best_SVR, x_col, y_col)
    
#     st.subheader('3D Plot')
#     fn_3Dplot2(Grid_df, Exp_Month, x_col, y_col)


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Load the saved model and scalers
best_SVR = joblib.load('best_SVR_model.pkl')
scalary = joblib.load('scalary.pkl')
scalarx = joblib.load('scalarx.pkl')

# Load the data
Corp_W_SF = pd.read_csv("Corp_W_SF_data.csv")

# Define the columns used in the model
feature_columns_NLM = ['GMRate', 'DMCoupon', 'DMPromo', 'DMOther', 'CouponPenetration', 'PromoPenetration', 'OtherPenetration', 'EconomicIndicator', 'SF']

# Define functions
def fn_GRID_Data_6vars(pmin, pmax, dmMin, dmMax, step_size, selected_values, bestNLM, x_col1, x_col2):
    x_values = np.arange(pmin, pmax, step_size)
    y_values = np.arange(dmMin, dmMax, step_size)
    A = pd.DataFrame({x_col1: x_values})
    B = pd.DataFrame({x_col2: y_values})
    A['key'] = 1
    B['key'] = 1
    df = pd.merge(A, B).drop('key', axis=1)
    
    for col, value in selected_values.items():
        df[col] = value
    
    df_NLM = df[feature_columns_NLM]
    scaled_df_NLM = scalarx.transform(df_NLM)
    dfpreds_NLM = bestNLM.predict(scaled_df_NLM)
    dfpreds_NLM_rescaled = scalary.inverse_transform(pd.DataFrame(dfpreds_NLM))
    df['Estimated Profit'] = dfpreds_NLM_rescaled
    return df

def fn_3Dplot2(df, Exp_Month, x_col1, x_col2):
    fig = go.Figure(data=[go.Scatter3d(
        x=df[x_col1],
        y=df[x_col2],
        z=df['Estimated Profit'],
        mode='markers',
        marker=dict(size=5, color=df['Estimated Profit'], colorscale='Viridis', opacity=0.8),
        name='Estimated_Profit'
    )])
    x1 = np.round(Exp_Month[x_col1].iloc[0], 2)
    y1 = np.round(Exp_Month[x_col2].iloc[0], 2)
    z1 = np.round(Exp_Month['Avg_Profit'].iloc[0], 2)
    tit = Exp_Month['month_of_year'].iloc[0]
    fig.add_trace(go.Scatter3d(
        x=[x1], y=[y1], z=[z1],
        mode='markers',
        marker=dict(size=10, color='red', symbol='circle'),
        name=tit+"("+str(x1)+","+str(y1)+","+str(z1)+")"
    ))
    fig.update_layout(
        title=f'3D Plot of Estimated Profit vs {x_col1} and {x_col2}',
        scene=dict(xaxis_title=x_col1, yaxis_title=x_col2, zaxis_title='Estimated Profit')
    )
    st.plotly_chart(fig)



def optim_plot2D2(df, fixed_value, ML_MODEL, x_col1, x_col2):
    df_p = df[np.isclose(df[x_col2], fixed_value)]
    plt.figure(figsize=(10, 4))
    x = df_p[x_col1]
    y = df_p['Estimated Profit']
    plt.plot(x, y)
    plt.xlabel(x_col1)
    plt.ylabel('Estimated Profit')
    plt.title(f"Estimated Profit vs {x_col1} when {x_col2}={fixed_value} ({ML_MODEL.__class__.__name__})")
    st.pyplot(plt)


# Streamlit app
st.title('Profit Prediction Analysis')
month = st.selectbox('Select Month-Year', Corp_W_SF['month_of_year'].unique())

# Define the list of columns
feature_columns_NLM1 = ['DMCoupon', 'DMPromo', 'DMOther', 'CouponPenetration', 'PromoPenetration', 'OtherPenetration']

# Create select boxes for dynamic column selection
x_col1 = st.selectbox('Select first column for grid', feature_columns_NLM1)
x_col2 = st.selectbox('Select second column for grid', feature_columns_NLM1)

st.header('Input Values for Grid')
pmin = st.number_input(f'Enter minimum {x_col1} value', key='pmin', min_value=None, max_value=None, step=0.005)
pmax = st.number_input(f'Enter maximum {x_col1} value', key='pmax', min_value=None, max_value=None, step=0.005)
dmMin = st.number_input(f'Enter minimum {x_col2} value', key='dmMin', min_value=None, max_value=None, step=0.005)
dmMax = st.number_input(f'Enter maximum {x_col2} value', key='dmMax', min_value=None, max_value=None, step=0.005)
step_size = st.number_input('Enter step size', format="%.5f", value=0.005, step=0.005)


# Selection for X-axis column in 2D plot
x_axis_column = st.selectbox('Select the column to use for the X-axis in the 2D plot', [x_col1, x_col2])

if st.button('Analyze'):
    Exp_Month = Corp_W_SF.loc[Corp_W_SF['month_of_year'] == month, feature_columns_NLM + ['Avg_Profit', 'month_of_year']]
    
    # Dictionary of constant values for other features
    selected_values = {col: np.round(Exp_Month[col].iloc[0], 2) for col in feature_columns_NLM if col not in [x_col1, x_col2]}
    
    Grid_df = fn_GRID_Data_6vars(pmin, pmax, dmMin, dmMax, step_size, selected_values, best_SVR, x_col1, x_col2)
    
    st.subheader('2D Plot')
    fixed_value_column = x_col2 if x_axis_column == x_col1 else x_col1

    # 2D Plot
    fixed_value = round(Exp_Month[fixed_value_column].iloc[0], 2)
    optim_plot2D2(Grid_df, fixed_value, best_SVR, x_axis_column, fixed_value_column)
    
    
    st.subheader('3D Plot')
    fn_3Dplot2(Grid_df, Exp_Month, x_col1, x_col2)
