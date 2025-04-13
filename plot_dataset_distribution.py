import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import random

# Setup
base_dir = os.getcwd()
output_dir = os.path.join(base_dir, 'image_data')
splits = ['train_images', 'valid_images', 'test_images']
label_mapping = {
    'gl': 'Glioma',
    'me': 'Meningioma',
    'pi': 'Pituitary',
    'no': 'No Tumor'
}

# Count images per class per split + store sample paths
counts = {split: {label: 0 for label in label_mapping.values()} for split in splits}
image_paths = {label: [] for label in label_mapping.values()}

for split in splits:
    split_path = os.path.join(output_dir, split)
    for file in os.listdir(split_path):
        if file.endswith('.jpg'):
            short_label = file.split('_')[0]
            full_label = label_mapping.get(short_label)
            if full_label:
                counts[split][full_label] += 1
                image_paths[full_label].append(os.path.join(split_path, file))


# Create DataFrame
data = [
    {"Split": split, "Label": label, "Count": count}
    for split, split_counts in counts.items()
    for label, count in split_counts.items()
]
df = pd.DataFrame(data)
df_pivot = df.pivot(index='Label', columns='Split', values='Count')
df_proportions = df_pivot.div(df_pivot.sum(axis=1), axis=0)

print("\nDataFrame with counts:")
print(df)


# Plot 1: Stacked bar chart with percentage labels
ordered_cols = ['train_images', 'valid_images', 'test_images']
df_proportions = df_proportions[ordered_cols]

ax = df_proportions.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20c')
plt.title('Proportion of Train/Validation/Test per Tumor Type')
plt.ylabel('Proportion')
plt.xlabel('Tumor Type')

# Add percentage labels
for x, (_, row) in enumerate(df_proportions.iterrows()):
    cumulative = 0
    for col in ordered_cols:
        value = row[col]
        ax.text(
            x=x,
            y=cumulative + value / 2,
            s=f"{value * 100:.0f}%",
            ha='center', va='center',
            fontsize=8, color='white', fontweight='bold'
            )
        cumulative += value

ax.legend(title='Split', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.show()


# Plot 2: Display a sample MRI image per class
fig, axs = plt.subplots(1, len(label_mapping), figsize=(12, 4))

for ax, (label, paths) in zip(axs, image_paths.items()):
    if paths:
        img = cv2.imread(random.choice(paths))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis('off')

plt.suptitle('Sample MRI Image per Tumor Class', fontsize=14)
plt.tight_layout()
plt.show()
