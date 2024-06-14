import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay
import shap
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

def model_performance(model_name, x_train_data, y_train_data, x_test_data, y_test_data):
    train_preds = model_name.predict(x_train_data)
    test_preds = model_name.predict(x_test_data)
    
    train_report = classification_report(y_train_data, train_preds)
    test_report = classification_report(y_test_data, test_preds)
    
    train_score = round(model_name.score(x_train_data, y_train_data), 4)
    test_score = round(model_name.score(x_test_data, y_test_data), 4)
    
    st.markdown('#### Classification report for training data')
    st.text(train_report)
    st.write('\n')
    st.write('#### Classification report for testing data')
    st.text(test_report)
    st.write('\n')
    st.markdown(f'#### The model score for training data is {train_score}')
    st.markdown(f'#### The model score for testing data is {test_score}')
    st.write('\n')

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Confusion Matrix for Testing Data", "Confusion Matrix for Training Data"))

    test_cm = confusion_matrix(y_test_data, test_preds)
    train_cm = confusion_matrix(y_train_data, train_preds)

    fig.add_trace(
        go.Heatmap(
            z=test_cm,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale="Viridis",
            showscale=False,
            text=test_cm,
            texttemplate="%{text}"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=train_cm,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale="Viridis",
            showscale=False,
            text=train_cm,
            texttemplate="%{text}"
        ),
        row=1, col=2
    )
    fig.update_layout(title_text="Confusion Matrices", height=600, width=1200)
    # fig.update_layout(title_text="Confusion Matrices")
    st.plotly_chart(fig)

def roc_score_auc_curve(model_name, x_train_data, y_train_data, x_test_data, y_test_data):
    train_auc = round(roc_auc_score(y_train_data, model_name.predict_proba(x_train_data)[:,1]), 4)
    test_auc = round(roc_auc_score(y_test_data, model_name.predict_proba(x_test_data)[:,1]), 4)
    st.markdown(f'#### AUC Score for Model on Training Data is {train_auc}')
    st.markdown(f'#### AUC Score for Model on Testing Data is {test_auc}')
    
    train_fpr, train_tpr, _ = roc_curve(y_train_data, model_name.predict_proba(x_train_data)[:,1])
    test_fpr, test_tpr, _ = roc_curve(y_test_data, model_name.predict_proba(x_test_data)[:,1])
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=train_fpr, y=train_tpr, mode='lines', name='Training', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=test_fpr, y=test_tpr, mode='lines', name='Testing', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='No Skill'))

    fig.update_layout(title='ROC Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      height=1000,
                      width=1500,
                      xaxis=dict(
                        tickfont=dict(size=20), 
                        titlefont=dict(size=24)),
                      yaxis=dict(
                            tickfont=dict(size=20),
                            titlefont=dict(size=24)))
    
    st.plotly_chart(fig)


# def k_fold_cross_valscore(model_name, x_train_data, y_train_data, folds):
#     metrics = ['recall', 'accuracy', 'precision', 'f1']
#     scores = {metric: cross_val_score(model_name, x_train_data, y_train_data, cv=folds, scoring=metric, verbose=0)
#               for metric in metrics}
    
#     cross_val_data = pd.DataFrame(scores)
    
#     for metric in metrics:
#         st.write(f"The mean {metric} for the model after {folds} folds is {np.mean(scores[metric]):.4f}")
    
#     st.write(cross_val_data)

# def plot_shap_summary(model, X_test):
#     explainer = shap.Explainer(model)
#     shap_values = explainer(X_test)
    
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
#     plt.tight_layout()
#     st.pyplot(fig)


# def plot_partial_dependence(model, X_test):
#     features = ['CashbackAmount', 'WarehouseToHome', 'OrderAmountHikeFromlastYear',
#                 'Tenure', 'DaySinceLastOrder', 'SatisfactionScore',
#                 'NumberOfDeviceRegistered', 'CouponUsed', 'OrderCount',
#                 'Gender_Male', 'CityTier','PreferredPaymentMode_DC']

#     fig, ax = plt.subplots(nrows=len(features)//3 + 1, ncols=3, figsize=(18, len(features) * 2))
    
#     display = PartialDependenceDisplay.from_estimator(model, X_test, features, grid_resolution=50, ax=ax.flatten()[:len(features)])
    
#     for i, axi in enumerate(ax.flatten()[:len(features)]):
#         axi.set_xlabel('Feature value')
#         axi.set_ylabel('Partial dependence')
#         axi.set_title(f'Partial Dependence of {features[i]}')

#     plt.tight_layout()
#     st.pyplot(fig)

def plot_calibration_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='markers+lines', name='Model'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Perfectly calibrated'))

    fig.update_layout(title='Caliberation Curve',
        xaxis_title='Predicted probability',
        yaxis_title='True probability',
        height=700,
        width=1500,
        xaxis=dict(
            tickfont=dict(size=20), 
            titlefont=dict(size=24)
        ),
        yaxis=dict(
            tickfont=dict(size=20),
            titlefont=dict(size=24)
        )
    )

    st.plotly_chart(fig)

def plot_learning_curve(model, X_train, y_train):
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='recall',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, mode='lines+markers', name='Training Score', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=train_sizes, y=val_scores_mean, mode='lines+markers', name='Cross-Validation Score', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean + train_scores_std, mode='lines', name='Train Score + STD', line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean - train_scores_std, mode='lines', name='Train Score - STD', line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=train_sizes, y=val_scores_mean + val_scores_std, mode='lines', name='Val Score + STD', line=dict(dash='dash', color='green')))
    fig.add_trace(go.Scatter(x=train_sizes, y=val_scores_mean - val_scores_std, mode='lines', name='Val Score - STD', line=dict(dash='dash', color='green')))

    fig.update_layout(title='Learning Curve',
                      xaxis_title='Training Examples',
                      yaxis_title='Score',
                      height=700,
                      width=1500,
                      xaxis=dict(
                        tickfont=dict(size=20), 
                        titlefont=dict(size=24)),
                      yaxis=dict(
                            tickfont=dict(size=20),
                            titlefont=dict(size=24)))

    st.plotly_chart(fig)
