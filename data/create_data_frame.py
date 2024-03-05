import pandas as pd

annotation_file = "new_annotations"
df = pd.read_json(annotation_file)
df = df[["patient_name", "flags"]]

# Initialize an empty DataFrame with the desired columns
result_df = pd.DataFrame(
    columns=["patient_name", "fp_1", "fp_2", "fp_3", "ff_1", "ff_2", "ff_3"]
)

# Iterate through each row of the original DataFrame
for index, row in df.iterrows():
    # Initialize a dictionary to store the counts for each flag
    flag_counts = {"fp_1": 0, "fp_2": 0, "fp_3": 0, "ff_1": 0, "ff_2": 0, "ff_3": 0}

    # Iterate through each flag in the 'flags' column
    for flag in row["flags"]:
        # Check if the flag is 'no_error'
        if flag == "no_error":
            flag_counts = {
                "fp_1": 0,
                "fp_2": 0,
                "fp_3": 0,
                "ff_1": 0,
                "ff_2": 0,
                "ff_3": 0,
            }
        else:
            # Extract the prefix and number from the flag
            parts = flag.split("_")
            prefix = parts[0]
            numbers = parts[-1]
            digits = [int(digit) for digit in numbers]

            # Update the counts in the dictionary based on the prefix and number
            if prefix == "fp":
                for num in numbers:
                    flag_counts[f"fp_{num}"] = 1
            elif prefix == "ff":
                for num in numbers:
                    flag_counts[f"ff_{num}"] = 1

    # Convert the flag_counts dictionary to a DataFrame
    flag_counts_df = pd.DataFrame([flag_counts])

    # Concatenate the patient_name column from the original DataFrame
    flag_counts_df["patient_name"] = row["patient_name"]

    # Reorder columns for better readability
    flag_counts_df = flag_counts_df[
        ["patient_name", "fp_1", "fp_2", "fp_3", "ff_1", "ff_2", "ff_3"]
    ]

    # Append the flag counts DataFrame as a new row to the result DataFrame
    result_df = pd.concat([result_df, flag_counts_df], ignore_index=True)

# Print the resulting DataFrame
print(result_df)

# Print the resulting DataFrame
# print(result_df)
result_df.to_csv("data_new.csv", index=False)

ax = result_df[["fp_1", "fp_2", "fp_3", "ff_1", "ff_2", "ff_3"]].sum().plot(kind="bar")
for i in ax.containers:
    ax.bar_label(i, label_type="edge")
ax.set_xlabel("Error Type")
ax.set_ylabel("No of images (key_frames)")
ax.imshow()
