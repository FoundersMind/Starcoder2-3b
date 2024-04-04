import pandas as pd

# Read the original CSV file
df = pd.read_csv("prompts.csv")

# Create a new DataFrame with "act" and "prompt" in separate columns
new_df = pd.DataFrame({'act': df['act'], 'prompt': df['prompt']})

# Save the new DataFrame to a new CSV file with a delimiter between "act" and "prompt"
new_df.to_csv("prompts_modified.csv", sep='\t', index=False)
