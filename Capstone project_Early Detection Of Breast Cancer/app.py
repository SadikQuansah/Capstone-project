import dash
from dash import dcc
from dash import html
from dash.dash_table.Format import Group
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from dash.dependencies import Input, Output
import pickle
import joblib


# Data from the first part of the project
output_csv_file_path = 'C:\\Users\\rexsa\\Desktop\\Capstone project_Early Detection Of Breast Cancer-20230905T062247Z-001\\Capstone project_Early Detection Of Breast Cancer\\Updated_Breast_Cancer.csv'
df = pd.read_csv(output_csv_file_path)

# Data from the second part of the project
file_path = 'C:\\Users\\rexsa\\Desktop\\Capstone project_Early Detection Of Breast Cancer-20230905T062247Z-001\\Capstone project_Early Detection Of Breast Cancer\\selected_data.csv'
df3 = pd.read_csv(file_path)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Data taken from each model Accuracy,Precision_Class_0,Precision_Class_1,Recall_Class_0,Recall_Class_1,F1_Score_Class_0,F1_Score_Class_1
data = {
    'Model': ['Random Forest', 'MLP', 'Decision Tree', 'SVM'],
    'Accuracy': [0.78, 0.83, 0.76, 0.81],
    'Precision_Class_0': [0.82, 0.84, 0.83, 0.81],
    'Precision_Class_1': [0.53, 0.73, 0.47, 0.85],
    'Recall_Class_0': [0.91, 0.96, 0.86, 0.99],
    'Recall_Class_1': [0.34, 0.38, 0.40, 0.22],
    'F1_Score_Class_0': [0.87, 0.90, 0.85, 0.89],
    'F1_Score_Class_1': [0.41, 0.50, 0.43, 0.35]
}

df2 = pd.DataFrame(data)


# Load the trained MLP model from the file
mlp_model = joblib.load('mlp_model.joblib')

#--------------------------------------------------------------------------------------------------------------------------------------

app = dash.Dash(__name__)

# Load the trained MLP model from the file
mlp_model = joblib.load('mlp_model.joblib')

output_csv_file_path = 'C:\\Users\\rexsa\\Desktop\\Capstone project_Early Detection Of Breast Cancer-20230905T062247Z-001\\Capstone project_Early Detection Of Breast Cancer\\Updated_Breast_Cancer.csv'
df = pd.read_csv(output_csv_file_path)

# Define target variable (higher risk or not)
df['Higher_Risk'] = ((df['Hazard Ratio(95% CI)'] > 1) & (df['Breast Cancer Occurences'] >= 1) & (df['P-value'] <= 0.05)).astype(int)


# Select relevant features and target variable
features = [
    'Cumulative Risk by Age, % (95% CI) 40 years old groups',
    'Cumulative Risk by Age, % (95% CI) 50 years old groups',
    'Cumulative Risk by Age, % (95% CI) 60 years old groups',
    'Cumulative Risk by Age, % (95% CI) 70 years old groups',
    'No.of Women', 'No.of Person-Years', 'No.of Events'
]
X = df[features]
y = df['Higher_Risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a dictionary to store algorithm names and their accuracy scores
algorithm_scores = {}

# Define the algorithms you want to compare
algorithms = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'MLP Classifier': MLPClassifier(max_iter=1000, random_state=42)
}

# Loop through each algorithm and calculate accuracy scores
for algorithm_name, algorithm in algorithms.items():
    # Preprocess the data with StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create an imputer instance for handling missing values (e.g., impute with mean)
    imputer = SimpleImputer(strategy='mean')

    # Fit and transform the imputer on your training data
    X_train_scaled_imputed = imputer.fit_transform(X_train_scaled)

    # Transform the testing data using the same imputer
    X_test_scaled_imputed = imputer.transform(X_test_scaled)

    # Train the algorithm
    algorithm.fit(X_train_scaled_imputed, y_train)

    # Make predictions
    y_pred = algorithm.predict(X_test_scaled_imputed)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100  # Convert accuracy to percentage

    # Store the accuracy score
    algorithm_scores[algorithm_name] = accuracy
#-----------------------------------------------------------------------------------------------------------------------------------
# Create the "Higher_Risk" column based on conditions
df['Higher_Risk'] = ((df['Hazard Ratio(95% CI)'] > 1) & (df['Breast Cancer Occurences'] >= 1) & (df['P-value'] < 0.05)).astype(int)

# Define the features you want to visualize
features = [
    'Cumulative Risk by Age, % (95% CI) 40 years old groups',
    'Cumulative Risk by Age, % (95% CI) 50 years old groups',
    'Cumulative Risk by Age, % (95% CI) 60 years old groups',
    'Cumulative Risk by Age, % (95% CI) 70 years old groups',
]

# Create a DataFrame with the selected features and the "Higher_Risk" variable
df_selected = df[features + ['Higher_Risk']]

# Group the data by the "Higher_Risk" variable and calculate the mean (percentage) in each category
grouped_data = df_selected.groupby(['Higher_Risk']).mean()

# Transpose the DataFrame for plotting
grouped_data = grouped_data.T.reset_index()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
border_style = {
    'border': '2px solid #FF5733',  # Border style (2px width, black color)
    'background-color': '#FFFFE0',
    'padding': '10px',  # Padding to create space between border and content
}
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
markdown_text = """
Summary:

Our analysis encompassed two significant phases, each with its unique dataset size, shedding light on the pivotal role data volume plays in predictive modeling. 

In Part 1, we confronted the challenges posed by a limited dataset. The smaller data pool resulted in the risk of overfitting, as the models learned to accommodate noise and fluctuations. While the models achieved high accuracy on the training data, their performance on unseen data was questionable.

Part 2 marked a transition to a more extensive dataset, and this change brought noteworthy improvements. With a larger and more diverse dataset, our predictive models became more robust and capable of making reliable predictions that could generalize to real-world situations.

This shift from a constrained dataset in Part 1 to a substantial one in Part 2 underscores the critical role data volume plays in the accuracy and efficacy of predictive models. It emphasizes the need for comprehensive and well-structured datasets to attain meaningful insights and precise predictions.

In summary, our analysis serves as a reminder of the significance of data quality and quantity in predictive modeling endeavors and highlights the necessity of leveraging abundant, high-quality data to achieve meaningful results in the field of machine learning and predictive analytics.
""" 
# Load the trained model
with open('svm_model.joblib', 'rb') as model_file:
    model = pickle.load(model_file)



app.layout = html.Div ([
    html.H1(" Breast Cancer: Early Detection System"),
    dcc.Markdown('The project is divided into two parts: part 1 focuses on predicting higher-risk groups for breast cancer, while part 2 involves forecasting chemotherapy utilization among different age groups for breast cancer treatment.'
        , style={'margin-top': '20px'}
    ),

    html.H1("Part 1:"),

    html.H2("Breast Cancer Analysis"),
    
    
    dcc.Markdown('''
    **Hazard Ratio (HR):**
    - HR helps us see if one group is more likely to have an event like breast cancer compared to another group.
    - If HR is 1 or more, it means the first group is at a higher risk.
    - For breast cancer, it helps us check if people with or without a family history have different risks.
    '''),

    dcc.Markdown('''
    **Breast Cancer Occurrence:**
    - This term means if someone is diagnosed with breast cancer.
    - We look at it based on a person's age and whether they have a family history.
    - It helps us see how age and family history link to breast cancer.
    '''),

    dcc.Markdown('''
    **P-value:**
    - P-value helps us know if our results are strong or just by chance.
    - If p≤ 0.05, it means we have strong proof for our results.
    - For breast cancer research, it helps us confirm if things like family history truly affect the risk.
    '''),

    html.H2("Understanding Relevant Features"),

    dcc.Markdown('''
    **Relevant Features Simplified:**

    Relevant features are like puzzle pieces that actually fit into the picture you're trying to create. They're not just random pieces; they are the ones that really help you see the whole picture clearly. These pieces, or features, are important because they give you clues about the thing you're trying to figure out or predict.

    Imagine you're trying to predict whether someone will win a race. Relevant features might include things like how fast they can run, how much they practice, and their past race results. These details matter because they help you make a good guess about who will win.

    On the other hand, things like the color of their shoes or the brand of their water bottle probably won't help you make a good prediction. Those would be like puzzle pieces that don't really fit into your race prediction picture.

    So, **relevant features** are like the important clues that help you understand or predict something better, and they're not just random bits of information.
    '''),
    dcc.Markdown('''
    In our analysis, I used the following features to understand and predict breast cancer occurrence:

    - Cumulative Risk by Age, % (95% CI) 40 years old groups
    - Cumulative Risk by Age, % (95% CI) 50 years old groups
    - Cumulative Risk by Age, % (95% CI) 60 years old groups
    - Cumulative Risk by Age, % (95% CI) 70 years old groups
    - No.of Women
    - No.of Person-Years
    - No.of Events

    These features provided valuable information to help us make predictions and gain insights into breast cancer risks.
    '''),
    html.H1("Stacked Bar Plot of Features by Risk Level (Percentage)"),

    dcc.Markdown('''
    The visual shows:
    
    - Different factors related to breast cancer risk on the horizontal axis.
    - How likely each factor is associated with a higher risk (orange) or lower risk (light blue) on the vertical axis.
    - By comparing the bar heights for each factor, you can see which factors are more strongly linked to higher or lower breast cancer risk. Orange bars mean higher risk, and light blue bars mean lower risk.
    
    In simpler terms, it helps you see which things might increase or decrease the risk of breast cancer.
    '''),

    dcc.Graph(id='stacked-bar-plot'),

    dcc.Markdown('''
    **Four Predictive Models Employed:**

    In our analysis, we utilized four predictive models to understand and predict breast cancer occurrence. These models played a crucial role in making predictions and evaluating the factors contributing to breast cancer risks.

    The four predictive models used were:

    1. Multi-Layer Perceptron (MLP)
    2. Random Forest 
    3. Support Vector Machine (SVM)
    4. Decision Tree

    Each of these models offered insights and helped us gain a better understanding of breast cancer risks.
    '''),

    html.H1("Algorithm Comparison with StandardScaler and Imputation "),
    dcc.Graph(id ='algorithm-comparison'),

    dcc.Markdown("""Conclusion: 
                *The high accuracy of the Decision Tree model on your dataset is a good thing.However, there's a problem called "overfitting" to be aware of. Overfitting occurs when the model gets too good at learning the training data, even the noisy or random parts. So, it performs really well on the training data but might not do as well with new data it hasn't seen before, like a test set or real-world information.*
                """),

    dcc.Markdown("In part 2, we will explore whether there are any differences or improvements by utilizing a larger dataset"),

    html.H1("Part 2:"),

    html.H2("Chemotherapy analysis"),

    
    html.H2("Exploring Chemotherapy Administration and Patient Factors"),

    dcc.Markdown('''
    **Analyzing Chemotherapy Trends by Age Group:**
    
    In our analysis, we looked at how certain features relate to the use of chemotherapy as a breast cancer treatment. Here's a straightforward explanation of each feature:

    **Age at Diagnosis**:
    - *Why it matters*: We wanted to see if a patient's age when diagnosed affects the likelihood of receiving chemotherapy.

    **Tumor Size**:
    - *Why it matters*: We explored whether having a larger tumor influences the decision to use chemotherapy.

    **Hormone Therapy**:
    - *Why it matters*: We investigated if patients receiving hormone therapy also tend to receive chemotherapy and under what circumstances.

    By examining these features alongside chemotherapy usage, we aimed to understand the factors influencing chemotherapy trends in breast cancer treatment.
    '''),

    dcc.Markdown('''
    **Four Predictive Models Employed:**
    In our analysis, we delved into the trends of chemotherapy utilization among breast cancer patients across different age groups. To achieve this, we employed four predictive models, each contributing valuable insights into the factors influencing chemotherapy decisions in breast cancer treatment.

    The four predictive models used were:

    1. Multi-Layer Perceptron (MLP)
    2. Random Forest 
    3. Support Vector Machine (SVM)
    4. Decision Tree

    These models were instrumental in shedding light on the complex interplay between age, tumor size, and hormone therapy in determining chemotherapy outcomes.
    '''),

    html.H1('Age Distribution by Chemotherapy (Box Plot)'),
    dcc.Graph(
        id='box-plot',
        figure={
            'data': [
                go.Box(
                    x=df3['chemotherapy'],
                    y=df3['age_at_diagnosis'],
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    name='Age at Diagnosis',
                    marker=dict(
                        color='blue'
                    )
                )
            ],
            'layout': go.Layout(
                xaxis={'title': 'Chemotherapy (0: No, 1: Yes)'},
                yaxis={'title': 'Age at Diagnosis'},
                boxmode='group',
                title='Age Distribution by Chemotherapy',
                xaxis_tickvals=[0, 1],
                xaxis_ticktext=['No Chemotherapy', 'Chemotherapy']
            )
        }
    ),
    dcc.Markdown('''Now that we have a visualization of this, let's compare it with predictive modeling to determine which predictive model will provide us with good accuracy.

'''),
    html.H1("Algorithm Comparison with StandardScaler and Imputation: "),

    dcc.Graph(id='bar-chart'),


    html.H1("Conclusion:"),
    dcc.Markdown(children=markdown_text),


    html.H1("Chemotherapy Prediction"),
    dcc.Input(id='age', type='number', placeholder='Age at Diagnosis'),
    dcc.Input(id='tumor_size', type='number', placeholder='Tumor Size'),
    dcc.Dropdown(id='hormone_therapy', options=[
        {'label': 'Yes', 'value': 1},
        {'label': 'No', 'value': 0}
    ], placeholder='Hormone Therapy'),
    html.Button('Predict', id='predict-button'),
    html.Div(id='prediction-output'),


    html.H1("Breast Cancer Prediction"),
    
    # Input fields for patient information
    dcc.Input(id='age', type='number', placeholder='Age'),
    dcc.Input(id='test_result', type='number', placeholder='Test Result'),
    dcc.Input(id='family_history', type='number', placeholder='Family History'),
    
    # Button to trigger prediction
    html.Button('Predict', id='predict-button'),
    
    # Display prediction result
    html.Div(id='prediction-output'),



    html.H1("Breast Cancer Prediction"),
    dcc.Input(id='age-input', type='number', placeholder='Age'),
    dcc.Input(id='tumor-size-input', type='number', placeholder='Tumor Size'),
    dcc.Dropdown(id='hormone-therapy-dropdown',
                 options=[
                     {'label': 'Hormone Therapy: Yes', 'value': 1},
                     {'label': 'Hormone Therapy: No', 'value': 0}
                 ],
                 placeholder='Hormone Therapy'),
    html.Button('Predict', id='predict-button'),
    html.Div(id='prediction-output')





])

@app.callback(
    Output('algorithm-comparison', 'figure'),
    Input('algorithm-comparison', 'relayoutData')
)
def update_graph(relayout_data):
    global algorithm_scores  # Make sure 'algorithm_scores' is a global variable

    # Initialize the 'figure' variable with a default value
    figure = {
        'data': [go.Bar(
            x=list(algorithm_scores.keys()),
            y=list(algorithm_scores.values()),
        )],
        'layout': go.Layout(
            title='Algorithm Comparison with StandardScaler and Imputation',
            xaxis=dict(title='Algorithms'),
            yaxis=dict(title='Accuracy (%)'),
            xaxis_tickangle=-45,
        )
    }

    # Check if 'relayout_data' is None
    if relayout_data is None:
        return figure  # Return the default 'figure' if 'relayout_data' is None

    # Check if 'xaxis.range' is in relayout_data
    if 'xaxis.range' in relayout_data:
        new_xaxis_range = relayout_data['xaxis.range']

        # Define a threshold for the number of bars displayed (e.g., 4)
        max_bars_displayed = 4

        # Check if the number of bars displayed exceeds the threshold
        if len(algorithm_scores) > max_bars_displayed:
            # Implement your logic here to update 'max_iter' and 'algorithm_scores'
            # Example: update_max_iter(new_xaxis_range)
            # Make sure to update 'algorithm_scores' accordingly

            return figure  # Return the 'figure' variable

@app.callback(
    Output('stacked-bar-plot', 'figure'),
    Input('stacked-bar-plot', 'relayoutData')
)
def update_graph(relayout_data):
    # Create the stacked bar plot using Plotly Express
    fig = px.bar(grouped_data, x='index', y=[0, 1], title="Stacked Bar Plot of Features by Higher Risk (Percentage)",
                labels={"index": "Features", "value": "Percentage"},
                color_discrete_sequence=['lightblue', 'orange'],  # Set the colors
                height=500)

    fig.update_xaxes(tickangle=45)
    
    # Update the legend labels
    fig.update_layout(legend_title_text='Risk Level', legend_traceorder='reversed')
    fig.update_traces(marker=dict(line=dict(width=0.5, color='Gray')))

    # Define custom legend labels
    custom_legend_labels = {0:'Low Risk', 1:'High Risk'}
    for i, label in custom_legend_labels.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=0, color='lightblue' if i == 0 else 'orange'), name=label))

    fig.update_layout(legend=dict(itemsizing='constant'))

    return fig

@app.callback(
    Output('bar-chart', 'figure'),
    Input('bar-chart', 'relayoutData')
)
def update_bar_chart(relayout_data):
    # Create a grouped bar chart
    fig = px.bar(
        df2,
        x='Model',
        y=['Accuracy', 'Precision_Class_0', 'Precision_Class_1', 'Recall_Class_0', 'Recall_Class_1'],
        title='Model Performance Metrics',
        labels={'value': 'Score'},
    )

    fig.update_layout(barmode='group')
    return fig

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('age-input', 'value'),
    Input('tumor-size-input', 'value'),
    Input('hormone-therapy-dropdown', 'value')
)
def make_prediction(n_clicks, age, tumor_size, hormone_therapy):
    if n_clicks is None:
        return ''
    
    # Preprocess input data
    input_data = pd.DataFrame({'age_at_diagnosis': [age],
                               'tumor_size': [tumor_size],
                               'hormone_therapy': [hormone_therapy]})
    
    # Standardize the input data (use the same scaler as during model training)
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Make predictions
    prediction = mlp_model.predict(input_data_scaled)[0]
    
    if prediction == 0:
        return 'Prediction: No Breast Cancer Risk'
    else:
        return 'Prediction: High Breast Cancer Risk'





if __name__ == '__main__':
 app.run_server(debug=True)
