import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer


def create_student_report(student_answers, question_metadata, df_question_avg_35, df_question_avg_4, output_file):
    # Process data and create the report table
    data = [["Question", "Your Answer", "Your Answer (Normalized)", "Average Human Answer", "Average GPT-3.5 Answer", "Average GPT-4 Answer"]]
    for i in range(1, 21):
        question = question_metadata.loc[question_metadata['QuestionNumber'] == i, 'Question'].values[0]
        q_short = question_metadata.loc[question_metadata['QuestionNumber'] == i, 'Q_Short'].values[0]
        answer = student_answers[str(i)]
        min_value = question_metadata.loc[question_metadata['QuestionNumber'] == i, 'Min'].values[0]
        max_value = question_metadata.loc[question_metadata['QuestionNumber'] == i, 'Max'].values[0]
        avg_gpt_4 = df_question_avg_4.loc[(df_question_avg_4['Q_Short'] == q_short), 'Answer'].iloc[0]
        avg_gpt_4 = round(avg_gpt_4, 2)

        avg_gpt_35 = df_question_avg_35.loc[(df_question_avg_4['Q_Short'] == q_short), 'Answer'].iloc[0]
        avg_gpt_35 = round(avg_gpt_35, 2)
        if answer != "NA":
            answer_normalized = round((float(answer) - min_value) / (max_value - min_value), 2)
            average_answer = round(student_answers_df[str(i)].apply(pd.to_numeric, errors='coerce').mean(), 2)
        else:
            answer_normalized = "NA"
            average_answer = "NA"

        data.append([question, answer, answer_normalized, average_answer, avg_gpt_35, avg_gpt_4])

    # Create the PDF report
    pdf = SimpleDocTemplate(output_file, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Noise Audit Report for {student_answers['Name']}", styles['Heading1']), Spacer(1, 20)]

    # Calculate column widths to fit the landscape A4 page
    first_col_width = 4 * inch
    other_col_widths = (landscape(A4)[0] - first_col_width) / 6

    for row in data:
        table_data = [[Paragraph(str(cell), styles['BodyText']) for cell in row]]
        table = Table(table_data, colWidths=[first_col_width] + [other_col_widths] * 3)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

    pdf.build(story)

# Read the CSV files
student_answers_df = pd.read_csv("student_answers_pii.csv")
question_metadata_df = pd.read_csv("questions-metadata.csv")

# Calculate mean for each question
question_metadata_df['Mean'] = student_answers_df.loc[:, '1':'20'].apply(pd.to_numeric, errors='coerce').mean()

json_data = stats.read_json_file("all-responses.json")
model_data = stats.extract_data(json_data)

columns = ['Model', 'Temperature', 'Iteration', 'Category', 'Question', 'Answer', 'Q_Short', 'Min', 'Max']
df_models = pd.DataFrame(model_data, columns=columns)

# group the filtered dataframe by Question and calculate the mean of Answer for each Question
df_question_avg_4 = df_models[df_models['Model'] == "gpt-4"].groupby('Q_Short')['Answer'].mean().reset_index()
df_question_avg_35 = df_models[df_models['Model'] == "gpt-3.5-turbo"].groupby('Q_Short')['Answer'].mean().reset_index()

# results_df = stats.normalize_and_analyze_data(df_models)

i = 0
# Generate a PDF report for each student
for _, student in student_answers_df.iterrows():
    i += 1
    output_file = "pdfs/"+f"{student['Name']}_report.pdf"
    create_student_report(student, question_metadata_df, df_question_avg_35, df_question_avg_4, output_file)
