import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def check_feature_values(df):
    # Dictionary to store validation results
    validation_results = {}

    # Feature 1: Patient Identification Number (patientid)
    if 'patientid' in df.columns:
        validation_results['patientid'] = {
            'Valid': all(df['patientid'].apply(lambda x: isinstance(x, (int, float)))),
            'Description': 'Number (Numeric)'
        }
    
    # Feature 2: Age (age)
    if 'age' in df.columns:
        validation_results['age'] = {
            'Valid': all(df['age'].apply(lambda x: isinstance(x, (int, float)))),
            'Description': 'In Years (Numeric)'
        }

    # Feature 3: Gender (gender)
    if 'gender' in df.columns:
        valid_genders = [0, 1]
        validation_results['gender'] = {
            'Valid': all(df['gender'].isin(valid_genders)),
            'Description': '1,0 (0 = female, 1 = male) (Binary)'
        }
    
    # Feature 4: Chest pain type (chestpain)
    if 'chestpain' in df.columns:
        valid_chestpain_values = [0, 1, 2, 3]
        validation_results['chestpain'] = {
            'Valid': all(df['chestpain'].isin(valid_chestpain_values)),
            'Description': '0,1,2,3 (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic) (Nominal)'
        }
    
    # Feature 5: Resting blood pressure (restingBP)
    if 'restingBP' in df.columns:
        validation_results['restingBP'] = {
            'Valid': all(df['restingBP'].between(94, 200)),
            'Description': '94-200 (in mm Hg) (Numeric)'
        }
    
    # Feature 6: Serum cholesterol (serumcholestrol)
    if 'serumcholestrol' in df.columns:
        validation_results['serumcholestrol'] = {
            'Valid': all(df['serumcholestrol'].between(126, 564)),
            'Description': '126-564 (in mg/dl) (Numeric)'
        }

    # Feature 7: Fasting blood sugar (fastingbloodsugar)
    if 'fastingbloodsugar' in df.columns:
        valid_fasting_blood_sugar_values = [0, 1]
        validation_results['fastingbloodsugar'] = {
            'Valid': all(df['fastingbloodsugar'].isin(valid_fasting_blood_sugar_values)),
            'Description': '0,1 > 120 mg/dl (0 = false, 1 = true) (Binary)'
        }
    
    # Feature 8: Resting electrocardiogram results (restingrelectro)
    if 'restingrelectro' in df.columns:
        valid_resting_ecg_values = [0, 1, 2]
        validation_results['restingrelectro'] = {
            'Valid': all(df['restingrelectro'].isin(valid_resting_ecg_values)),
            'Description': '0,1,2 (0: normal, 1: ST-T wave abnormality, 2: probable or definite left ventricular hypertrophy) (Nominal)'
        }
    
    # Feature 9: Maximum heart rate achieved (maxheartrate)
    if 'maxheartrate' in df.columns:
        validation_results['maxheartrate'] = {
            'Valid': all(df['maxheartrate'].between(71, 202)),
            'Description': '71-202 (Numeric)'
        }
    
    # Feature 10: Exercise induced angina (exerciseangia)
    if 'exerciseangia' in df.columns:
        valid_exercise_angina_values = [0, 1]
        validation_results['exerciseangia'] = {
            'Valid': all(df['exerciseangia'].isin(valid_exercise_angina_values)),
            'Description': '0,1 (0 = no, 1 = yes) (Binary)'
        }
    
    # Feature 11: Oldpeak = ST (oldpeak)
    if 'oldpeak' in df.columns:
        validation_results['oldpeak'] = {
            'Valid': all(df['oldpeak'].between(0, 6.2)),
            'Description': '0-6.2 (Numeric)'
        }
    
    # Feature 12: Slope of the peak exercise ST segment (slope)
    if 'slope' in df.columns:
        valid_slope_values = [1, 2, 3]
        validation_results['slope'] = {
            'Valid': all(df['slope'].isin(valid_slope_values)),
            'Description': '1,2,3 (1: upsloping, 2: flat, 3: downsloping) (Nominal)'
        }
    
    # Feature 13: Number of major vessels (noofmajorvessels)
    if 'noofmajorvessels' in df.columns:
        valid_major_vessels_values = [0, 1, 2, 3]
        validation_results['noofmajorvessels'] = {
            'Valid': all(df['noofmajorvessels'].isin(valid_major_vessels_values)),
            'Description': '0,1,2,3 (Numeric)'
        }
    
    # Feature 14: Classification (target)
    if 'target' in df.columns:
        valid_target_values = [0, 1]
        validation_results['target'] = {
            'Valid': all(df['target'].isin(valid_target_values)),
            'Description': '0,1 (0 = Absence of Heart Disease, 1 = Presence of Heart Disease) (Binary)'
        }
    
    return validation_results

def card_type(df, category_threshold=10, continuous_threshold=30):
    # First part: Prepare the dataset with cardinalities, % cardinality variation, and types
    df_temp = pd.DataFrame([df.nunique(), df.nunique()/len(df) * 100, df.dtypes])  # Cardinality and % variation of cardinality
    df_temp = df_temp.T  # Since it gives the column values as columns, transpose to have them as rows
    df_temp = df_temp.rename(columns={0: "Card", 1: "%_Card", 2: "Type"})  # Rename after transposition for clarity

    # Correction for when there's only one value
    df_temp.loc[df_temp.Card == 1, "%_Card"] = 0.00

    # Create the suggested variable type column, starting with all as categorical but could start with any, adjusting filters accordingly
    df_temp["suggested_type"] = "Categorical"
    df_temp.loc[df_temp["Card"] == 2, "suggested_type"] = "Binary"
    df_temp.loc[df_temp["Card"] >= category_threshold, "suggested_type"] = "Discrete numeric"
    df_temp.loc[df_temp["%_Card"] >= continuous_threshold, "suggested_type"] = "Continuous numeric"

    return df_temp

def get_CV(df, column):
    column = [column] if type(column) == str else column
    return df[column].describe().T["std"] / df[column].describe().T["mean"] * 100


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_health(data, best_xgb_model):
    # Convert the dictionary to DataFrame
    input_data = pd.DataFrame(data)
    
    # Predict using the best model
    prediction = best_xgb_model.predict(input_data)
    
    # Get the probabilities for each class (if the model supports it)
    probabilities = best_xgb_model.predict_proba(input_data)

    # Create a bar plot to visualize the probabilities
    labels = ['No heart disease', 'Heart disease']
    prob_values = probabilities[0]
    colors = ['#76C7C0', '#FF6F61']

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    barplot = sns.barplot(x=labels, y=prob_values, palette=colors)
    barplot.set(ylim=(0, 1))

    # Add labels to the bars
    for index, value in enumerate(prob_values):
        plt.text(index, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    # Add titles and labels
    plt.title('Heart Disease Probability', fontsize=16)
    plt.ylabel('Probability', fontsize=14)

    # Show the plot
    plt.show()

    # Based on the influencing features identified
    recommendations = []

    # Recommendations based on the values of the variables
    pain_type = input_data['chestpain'].values[0]
    if pain_type == 0:
        recommendations.append("Chest pain type 0 indicates typical angina. Consider a medical evaluation to determine the cause.")
    elif pain_type == 1:
        recommendations.append("Chest pain type 1 indicates atypical angina. Additional tests are recommended to rule out heart disease.")
    elif pain_type == 2:
        recommendations.append("Chest pain type 2 is non-anginal pain. It may be less severe, but still should be evaluated to rule out heart problems.")
    elif pain_type == 3:
        recommendations.append("Chest pain type 3 indicates asymptomatic. This is a good sign, but continue with regular check-ups to monitor cardiovascular health.")

    if input_data['restingBP'].values[0] > 140:
        recommendations.append("Your resting blood pressure is high and may indicate hypertension. It's important to reduce salt intake, improve your diet, and consider medication under medical supervision.")

    if input_data['serumcholestrol'].values[0] < 120:
        recommendations.append("Your serum cholesterol level is low. Ensure you maintain a balanced diet and consult a doctor to ensure your cholesterol level is appropriate.")
    elif input_data['serumcholestrol'].values[0] > 240:
        recommendations.append("High serum cholesterol levels may increase the risk of heart disease. Switch to a low-cholesterol diet and consult a physician.")

    if input_data['fastingbloodsugar'].values[0] == 1:
        recommendations.append("Elevated fasting blood sugar may be a sign of diabetes. Additional tests are recommended and adjust your diet and lifestyle according to medical advice.")

    if input_data['restingrelectro'].values[0] == 1:
        recommendations.append("Abnormalities in the electrocardiogram may indicate heart issues. Additional tests are recommended to assess heart health.")
    elif input_data['restingrelectro'].values[0] == 2:
        recommendations.append("Probable left ventricular hypertrophy requires specialized medical attention to determine appropriate treatment.")

    if input_data['maxheartrate'].values[0] > 170:
        recommendations.append("Your maximum heart rate is high. Consider a cardiovascular evaluation and adjust your exercise regimen accordingly.")

    if input_data['slope'].values[0] == 1:
        recommendations.append("An upsloping ST segment may indicate exercise-induced ischemia. A medical evaluation is recommended.")
    elif input_data['slope'].values[0] == 3:
        recommendations.append("A downsloping ST segment may indicate significant heart disease. A comprehensive medical evaluation is necessary.")

    if input_data['noofmajorvessels'].values[0] > 1:
        recommendations.append("The number of major vessels affected is high. It is crucial to follow an appropriate medical treatment plan and make lifestyle changes to manage coronary artery disease.")

    # Display recommendations
    print("Based on your profile, here are some recommendations to improve your health:")
    for rec in recommendations:
        print("- " + rec)