import csv

train_list = []
# Open the CSV file for reading
with open('/home/jayaram/research/NLP_course_papers_and_content/Assignments/Assign_2/train.csv', 'r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Use the len() function to count the rows and subtract 1 to exclude the header row if it exists
    # row_count = len(list(csv_reader)) - 1

    train_list = list(csv_reader)
    print('jai')

# Print the row count
print(f"Number of rows (excluding header if present): {row_count}")
