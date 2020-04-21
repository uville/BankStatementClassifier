import os
import io
from datetime import date

import pandas as pd
from textblob.classifiers import NaiveBayesClassifier


data = "data/training/SortedTransactions.csv"
new_transactions = "data/input/tapahtumat20200101-20200418.csv"
categories_path = "data/categories/categories.txt"
encoding = "iso-8859-1"
save_processed_transactions_to_excel = True
output_filename_excel = "ProcessedTransaction{0}.xlsx".format(str(date.today()))
output_path_excel = "data//output_excels//"


def __run__():
    __guess_class__()


def __init__(data):
    # If old data exists, read data to memory, else create empty dataframe
    if os.path.exists(data):
        prev_data = pd.read_csv(data, encoding=encoding, sep=',', decimal=".")
    else:
        prev_data = pd.DataFrame(columns=['date', 'desc', 'amount', 'cat'])

    return prev_data


def __readtransactions__(filename):
    # Read transaction data
    transactionData = pd.read_csv(filename, encoding=encoding, sep=';', decimal=",")

    # Join Column 'Laji' to 'Saaja/maksaja'
    transactionData["Saaja/Maksaja"] = transactionData["Laji"].map(str) + " " + transactionData["Saaja/Maksaja"].map(
        str)

    # Select only needed columns for dataframe
    df = transactionData.loc[:, ("Kirjauspäivä", "Määrä  EUROA", "Saaja/Maksaja")]

    # Rename Finnish column names to English
    df.rename(columns={"Kirjauspäivä": "date", "Saaja/Maksaja": "desc", "Määrä  EUROA": "amount"}, inplace=True)

    # Change data type to str
    df['date'].astype(str)
    df['desc'].astype(str)

    return df


def __extractor__(desc):
    # Split individual words to tokens
    tokens = desc.split(" ")

    # Init features
    features = {}

    # Iterate all tokens
    for token in tokens:
        if token == "":
            continue
        features[token] = True

    return features


def _get_training_(df):
    # Init new list
    train = []

    # Filter out empty categories
    subset = df[df['cat'] != '']

    for ind in subset.index:
        train.append((subset['desc'][ind], subset['cat'][ind]))

    return train


def _read_categories():
    # Init categories
    categories = {}

    print("\n")
    print("Categories available:")

    with io.open(categories_path, encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            categories[i] = line.strip()
            print(categories[i])

    print("\n")

    return categories


def _add_new_category(category):
    # Ask the name for new category, which will be added to categories.txt
    input_category_name = input("Type the name of new category: ")

    # Trim start and end of input
    input_category_name = input_category_name.strip()

    with io.open(categories_path, 'a', encoding='utf-8') as f:
        f.write('\n' + category + ", " + input_category_name)


def __guess_class__():
    # Read classified transactions to dataframe
    prev_data = __init__(data)

    # Read new transaction to dataframe
    df = __readtransactions__(new_transactions)

    # Read categories from categories.txt
    categories = _read_categories()

    # Init the classifier
    classifier = NaiveBayesClassifier(_get_training_(prev_data), __extractor__)

    # Add empty column for category
    df['cat'] = ""

    # Try to guess the category by iterating row by row
    print("Start processing transactions...")
    print("\n")
    for index, row in df.iterrows():

        print("Processing new transaction...")

        # Assign words from description column for the classifier
        stripped_text = row['desc']
        print("Tokens used for classificaton: " + stripped_text)

        # Guess a category using the classifier
        # If classifier is empty, guess nothing
        if len(classifier.train_set) > 1:
            guess = classifier.classify(stripped_text)
        else:
            guess = ""

        # Print transaction
        print("Transaction: " + str((row['date'], row['amount'], row['desc'])))
        print("Categories available: ")
        print(categories)
        print("ClassifierGuess: " + str(guess))
        print("\n")

        input_value = input("Confirm guess by pressing enter, otherwise type new category > ")

        if input_value.lower() == 'q':
            # If the input was 'q' then quit
            break
        if input_value == "":
            # If the input was blank then the guess was right
            df.loc[df.index[index], 'cat'] = guess
            classifier.update([(stripped_text, guess)])
        else:
            # Otherwise, the guess was wrong
            try:
                # Try converting the input to an integer category number
                # If it works then a valid category have been entered
                category_number = int(input_value)
                category = categories[category_number]
            except KeyError:
                # Otherwise, a new category entered, add it to list of categories
                category = input_value
                _add_new_category(category)
                categories = _read_categories()
                category = categories[category_number]

            # Add correct category to dataframe
            df.loc[df.index[index], 'cat'] = category
            # Update classifier
            classifier.update([(stripped_text, category)])

    # Remove rows without category
    df_without_empty_categories = df[df.cat != '']

    # Save transactions to Excel, if enabled
    if save_processed_transactions_to_excel:
        df_without_empty_categories.to_excel(output_path_excel + output_filename_excel)

    # Save data with category to training file
    prev_data = pd.concat([prev_data, df_without_empty_categories], sort=True)
    prev_data.to_csv(data, index=False, encoding=encoding)