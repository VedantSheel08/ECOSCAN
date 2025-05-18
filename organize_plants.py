import os
import shutil
import random

def count_and_organize_images(source_dir, target_count=400):
    # Count images in each category
    categories = ['Healthy', 'Powdery', 'Rust']
    counts = {}
    
    for category in categories:
        category_path = os.path.join(source_dir, category)
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            counts[category] = len(files)
            print(f"{category}: {len(files)} images")
            
            # If more than target_count images, randomly select target_count
            if len(files) > target_count:
                selected_files = random.sample(files, target_count)
                
                # Move non-selected files to a backup directory
                backup_dir = os.path.join(source_dir, f"{category}_backup")
                os.makedirs(backup_dir, exist_ok=True)
                
                for file in files:
                    if file not in selected_files:
                        shutil.move(
                            os.path.join(category_path, file),
                            os.path.join(backup_dir, file)
                        )
                print(f"Moved excess files from {category} to backup")
    
    return counts

# Path to the training data
train_dir = "Data/Plants/train/Train"

# Organize images (400 per category = 1200 total)
counts = count_and_organize_images(train_dir, 400)
print("\nFinal counts:", counts) 