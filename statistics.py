import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_data(json_data):
    questions_meta = pd.read_csv("questions-metadata.csv")
    
    data = []
    for item in json_data:
        model = item["model"]
        temperature = item["temperature"]
        iteration = item["iteration"]
        responses = item["responses"]
        
        for category, questions in responses.items():
            if category != "ranking_questions":
                for question_data in questions:
                    question = question_data["question"]
                    answer = question_data["answer"]
                    row = questions_meta.loc[questions_meta['Question'] == question]
                    q_short = questions_meta.at[row.index[0], 'Q_Short']
                    min = questions_meta.at[row.index[0], 'Min']
                    max = questions_meta.at[row.index[0], 'Max']
                    data.append((model, temperature, iteration, category, question, answer, q_short, min, max))
    return data

def extract_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    questions_meta = pd.read_csv("questions-metadata.csv")

    data = []
    for index, row in df.iterrows():
        model = "Human"
        temperature = float(row["Self Estimate"] / 10)
        iteration = index

        for qnumber in questions_meta['QuestionNumber']:
            question_row = questions_meta.loc[questions_meta['QuestionNumber'] == qnumber]
            question = question_row.at[question_row.index[0], 'Question']
            q_short = question_row.at[question_row.index[0], 'Q_Short']
            category = ""
            min_val = question_row.at[question_row.index[0], 'Min']
            max_val = question_row.at[question_row.index[0], 'Max']
            answer = row[str(qnumber)]
            data.append((model, temperature, iteration, category, question, answer, q_short, min_val, max_val))

    return data

def calculate_noise_audit(data):
    df = pd.DataFrame(data, columns=["Model", "Temperature", "Iteration", "Category", "Question", "Answer", "Q_Short", "Min", "Max"])
    results = df.groupby(["Model", "Temperature", "Question"]).agg(
        Mean=("Answer", "mean"),
        Median=("Answer", "median"),
        Variance=("Answer", "var"),
        STD=("Answer", "std"),
        Relative_Variance=("Answer", lambda x: x.var() / x.mean()**2)
    )
    results["Relative_Variance"] = results["Variance"]/results["Mean"]**2
    results["Relative_STD"] = results["STD"] / results["Mean"]
    return results

def save_noise_audit_to_csv(results, file_path):
    results.to_csv(file_path)


def remove_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

def normalize_and_analyze_data(df):
    # Remove outliers from 'Answer' column before normalization
    df = df[df['Answer'] != "NA"]
    df['Answer'] = df['Answer'].astype(float)
    df = remove_outliers(df, 'Answer')

    # Normalize the data for each question separately
    normalized_answers = []
    for _, row in df.iterrows():
        min_value = row['Min']
        max_value = row['Max']
        answer = row['Answer']
        normalized_answer = (answer - min_value) / (max_value - min_value)
        normalized_answers.append(normalized_answer)
    df['Normalized'] = normalized_answers

    df['Normalized'].fillna(df['Normalized'].mean(), inplace=True)

    # Calculate mean, median, mode, range, and standard deviation
    grouped_data = df.groupby(['Model', 'Temperature', 'Q_Short'])

    mean = grouped_data['Normalized'].mean()
    median = grouped_data['Normalized'].median()
    mode = grouped_data['Normalized'].agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else None)
    _range = grouped_data['Normalized'].agg(np.ptp)
    # std_deviation = grouped_data['Normalized'].std()
    std_deviation = grouped_data['Normalized'].agg(lambda x: x.std() if len(x) > 1 else 0)
    # Create a new DataFrame for the results
    results_df = pd.DataFrame({
        'Model': mean.index.get_level_values('Model'),
        'Temperature': mean.index.get_level_values('Temperature'),
        'Q_Short': mean.index.get_level_values('Q_Short'),
        'Mean': mean.values,
        'Median': median.values,
        'Mode': mode.values,
        'Range': _range.values,
        'Standard Deviation': std_deviation.values
    })

    return results_df

def perform_anova_test(data):
    print("ANOVA Test...")
    # Get unique questions
    unique_questions = data['Q_Short'].unique()
    
    # Create an empty dataframe for storing the ANOVA results
    anova_results = pd.DataFrame(columns=['Question', 'F_value', 'p_value'])
    
    # Perform ANOVA for each question
    for question in unique_questions:
        question_data = data[data['Q_Short'] == question]
        unique_temperatures = question_data['Temperature'].unique()
        
        # Prepare the data for ANOVA
        samples = [question_data[question_data['Temperature'] == temp]['Mean'] for temp in unique_temperatures]
        
        # Perform ANOVA
        f_value, p_value = stats.f_oneway(*samples)
        
        # Append the results to the dataframe
        anova_results = anova_results.append({
            'Question': question,
            'F_value': f_value,
            'p_value': p_value
        }, ignore_index=True)
        
    return anova_results

def visualize_anova_results(anova_results, significance_level=0.05):
    # Create a DataFrame from the ANOVA results and add a 'Significance' column
    anova_df = pd.DataFrame(anova_results, columns=['Question', 'F_value', 'p_value'])
    anova_df['Significance'] = anova_df['p_value'].apply(lambda x: 'Significant' if x <= significance_level else 'Not Significant')
    
    # Create a bar chart
    plt.figure(figsize=(12, 6))
    
    sns.barplot(data=anova_df, x='Question', y='p_value', hue='Significance', dodge=False)
    
    # Set the significance level line
    plt.axhline(y=significance_level, color='red', linestyle='--', label=f'Significance Level: {significance_level}')
    
    # Customize the chart
    plt.title('ANOVA Results: P-Values for Each Question')
    plt.xlabel('Question')
    plt.ylabel('P-Value')
    plt.legend()
    plt.xticks(rotation=90)
    plt.grid(linestyle='--')
    # Show the chart
    plt.show()
    
def plot_mean_scatter_plot(data, model):
    # Filter data by model
    model_data = data[data['Model'] == model]
    
    # Get unique temperatures and questions
    unique_temperatures = model_data['Temperature'].unique()
    unique_questions = model_data['Q_Short'].unique()
    
    # Set up the colormap
    colormap = plt.cm.get_cmap("coolwarm", len(unique_temperatures))
    
    # Plot the data
    fig, ax = plt.subplots(figsize=(18, 10))
    
    for i, temp in enumerate(unique_temperatures):
        temp_data = model_data[model_data['Temperature'] == temp]
        ax.scatter(temp_data['Q_Short'], temp_data['Mean'], color=colormap(i), label=f'Temperature: {temp}', alpha=0.6, edgecolors='w', s=100)
        
    # Customize the plot
    ax.set_xticks(np.arange(len(unique_questions)))
    ax.set_xticklabels(unique_questions, rotation='vertical', fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Questions')
    ax.set_ylabel('Mean Value (Normalized)')
    ax.set_title(f'Mean Answer Values for Model: {model}')
    ax.legend(loc='best', fontsize='small')
    
    # Show the plot
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.show()
    
def plot_std_scatter_plot(data, model):
    # Filter data by model
    model_data = data[data['Model'] == model]
    # for i, row in model_data.iterrows():
    #     print(i)
    #     print(row["Q_Short"])
    #     print(row["Temperature"])
    #     print(row["Standard Deviation"])
    # Get unique temperatures and questions
    unique_temperatures = model_data['Temperature'].unique()
    unique_questions = model_data['Q_Short'].unique()
    
    # Set up the colormap
    colormap = plt.cm.get_cmap("coolwarm", len(unique_temperatures))
    
    # Plot the data
    fig, ax = plt.subplots(figsize=(18, 10))
    
    for i, temp in enumerate(unique_temperatures):
        temp_data = model_data[model_data['Temperature'] == temp]
        ax.scatter(temp_data['Q_Short'], temp_data['Standard Deviation'], color=colormap(i), label=f'Temperature: {temp}', alpha=0.6, edgecolors='w', s=100)
        
    # Customize the plot
    ax.set_xticks(np.arange(len(unique_questions)))
    ax.set_xticklabels(unique_questions, rotation='vertical', fontsize=10)
    ax.set_ylim(0, 0.3)
    ax.set_xlabel('Questions')
    ax.set_ylabel('Standard Deviation')
    ax.set_title(f'Answer Standard Deviation Values for Model: {model}')
    ax.legend(loc='best', fontsize='small')
    
    # Show the plot
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.show()
    
def plot_std_line_chart(data, ignore_zero_std=False):
    xlabel='Temperature / Proficiency Estimate'
    ylabel='Average Standard Deviation'
    # Set plot style and size
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))
    
    if ignore_zero_std:
        data = data[data['Standard Deviation'] > 0]
    
    for model in data['Model'].unique():
        model_data = data[data['Model'] == model]
        model_data = model_data.groupby('Temperature')['Standard Deviation'].mean().reset_index()
    
        # Plot the line chart for the current model
        plt.plot(model_data['Temperature'], model_data['Standard Deviation'], label=model, marker='o')
    
        # Add data point numbers
        for i, row in model_data.iterrows():
            plt.text(row['Temperature'], row['Standard Deviation'], f'{row["Standard Deviation"]:.2f}', fontsize=10, ha='left', va='bottom')
            
        # Calculate and display the average value across each model
        avg_std = np.mean(model_data['Standard Deviation'])
        plt.axhline(avg_std, linestyle='--', linewidth=1, color=plt.gca().lines[-1].get_color())
        plt.text(model_data['Temperature'].max() + 0.1, avg_std, f'{avg_std:.3f}', fontsize=10, ha='left', va='center')
    
    # Customize the chart
    plt.title('Average Standard Deviation by Temperature for Each Model')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title='Model')
    
    # Show the chart
    plt.show()

def plot_mean_std_by_question(data, xlabel='Questions', ylabel='Standard Deviation of Mean'):
    # Set plot style and size
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(15, 6))
    
    # Iterate through unique models
    for model in data['Model'].unique():
        # Group data by model and question, then calculate the mean for each group
        model_data = data[data['Model'] == model]
        model_data_mean = model_data.groupby(['Q_Short'])['Standard Deviation'].mean().reset_index()
        
        # Get unique questions for plotting
        questions = model_data_mean['Q_Short'].unique()
        question_indices = np.arange(len(questions))
        
        # Plot the line chart for the current model
        plt.plot(question_indices, model_data_mean['Standard Deviation'], label=model, marker='o')
        
    # Customize the chart
    plt.title('Standard Deviation of Mean by Question for Each Model')
    plt.xticks(question_indices, questions, rotation='vertical')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show the chart
    plt.tight_layout()
    plt.show()

def plot_mean_std_by_question_bar_chart(data):
    xlabel='Questions'
    ylabel='Standard Deviation of Mean'
    # Set plot style and size
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(15, 6))
    
    models = data['Model'].unique()
    num_models = len(models)
    model_colors = plt.cm.get_cmap('viridis', num_models)
    
    # Group data by question
    question_data = data.groupby(['Q_Short', 'Model'])['Standard Deviation'].mean().reset_index()
    questions = question_data['Q_Short'].unique()
    question_indices = np.arange(len(questions))
    
    # Set bar width
    bar_width = 1 / (num_models + 1)
    
    # Iterate through unique models
    for i, model in enumerate(models):
        model_data = question_data[question_data['Model'] == model]
        plt.bar(question_indices + i * bar_width, model_data['Standard Deviation'], width=bar_width,
                label=model, color=model_colors(i))
        
    # Customize the chart
    plt.title('Standard Deviation of Mean by Question for Each Model')
    plt.xticks(question_indices + bar_width * (num_models - 1) / 2, questions, rotation='vertical')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show the chart
    plt.tight_layout()
    plt.show()

def analyze_and_draw_results():
    json_data = read_json_file("all-responses.json")
    model_data = extract_data(json_data)

    sa = extract_data_from_csv("student_answers_pii.csv")

    columns = ['Model', 'Temperature', 'Iteration', 'Category', 'Question', 'Answer', 'Q_Short', 'Min', 'Max']
    # Create a DataFrame from the extracted data
    df_model = pd.DataFrame(model_data, columns=columns)
    results_model = normalize_and_analyze_data(df_model)

    df_stu = pd.DataFrame(sa, columns=columns)
    results_stu = normalize_and_analyze_data(df_stu)

    results_df = pd.concat([results_model, results_stu])

    results_df.to_csv("statistical_analysis_results.csv", index=False)

    # visualize_anova_results(perform_anova_test(results_model))
    # visualize_anova_results(perform_anova_test(results_stu))
    plot_std_scatter_plot(results_df, "gpt-4")

    # plot_std_line_chart(results_df, ignore_zero_std=True)

    # plot_mean_std_by_question_bar_chart(results_df)

if __name__ == "__main__":
    analyze_and_draw_results()
