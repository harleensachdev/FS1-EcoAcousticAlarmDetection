import os
import torch
import pandas as pd
import torchaudio
from tqdm import tqdm

from config import (
    AUDIO_DIR,
    SPECTROGRAM_DIR,
    EVALUATEAUDIO_DIR,
    EVALUATEDATAPATH,
    METADATA_PATH,
    REQUIRED_CLASSES,
    N_WAY,
    N_SUPPORT,
    N_QUERY,
    TEST_SIZE,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS
)

def getmetadata():
    """
    1. Scan through audio directory
    2. Create/update metadata CSV
    """
    
    # Ensure metadata directory exists
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

    # Create metadata file if empty/not exist
    if not os.path.exists(METADATA_PATH) or os.path.getsize(METADATA_PATH) == 0:
        print("Creating a new metadata file")
        metadata_df = pd.DataFrame(columns=[
            'file_path', 'label', 'spectrogram_path', 'duration',
            'prediction_confidence', 'prediction', 'prediction_correct'
        ])
        metadata_df.to_csv(METADATA_PATH, index=False)
    else:
        # Load existing metadata file
        metadata_df = pd.read_csv(METADATA_PATH)
    
    # List out all audio files
    audio_files = []
    for root, _, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} total audio files in {AUDIO_DIR}")
    
    # Debug: Show some example paths
    if len(audio_files) > 0:
        print("Example file paths:")
        for i, path in enumerate(audio_files[:5]):
            print(f"  {path}")
        if len(audio_files) > 5:
            print(f"  ... and {len(audio_files) - 5} more files")
    else:
        print(f"WARNING: No audio files found in {AUDIO_DIR}")
        print("Expected directory structure:")
        print("  audio/train/alarm/*.wav")
        print("  audio/train/non_alarm/*.wav") 
        print("  audio/train/background/*.wav")
        print("  audio/train/highfreq_noise/*.wav")
        print("  audio/train/insect_call/*.wav")
        print("  audio/train/weather_rain/*.wav")
        print("  audio/train/lowfreq_noise/*.wav")
        print("  audio/test/[same classes]/*.wav (optional)")
        return metadata_df
    
    existing_files = set(metadata_df['file_path'].tolist() if 'file_path' in metadata_df.columns else [])
    
    # Find new files
    new_files = [f for f in audio_files if f not in existing_files]
    if len(new_files) == 0:
        print("No new audio files to process")
        return metadata_df
    else:
        print(f"Found {len(new_files)} new audio files to process")

    # Configure spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    # Process new files
    new_data = []
    files_by_class = {}
    
    for file_path in tqdm(new_files, desc="Processing audio files"):
        try:
            # Extract label from file path
            label = extract_label_from_path(file_path)
            if label is None:
                print(f"Warning: Could not extract label from {file_path}")
                continue

            # Count files by class for debugging
            if label not in files_by_class:
                files_by_class[label] = 0
            files_by_class[label] += 1

            # Load and process audio
            waveform, sr = load_audio(file_path)
            if waveform is None:
                print(f"Skipping {file_path} - could not load audio")
                continue

            # Pad or trim
            waveform = pad_or_trim(waveform)

            # Get duration
            duration = waveform.shape[1] / sr

            # Create the spectrogram
            spectrogram = mel_spectrogram(waveform)
            # Add small constant, take log
            spectrogram = torch.log(spectrogram + 1e-9)

            # Generate paths, ensure spectrogram dir exists
            spectrogram_path = generate_spectrogram_path(file_path)
            os.makedirs(os.path.dirname(spectrogram_path), exist_ok=True)

            # Save spectrogram
            torch.save(spectrogram, spectrogram_path)

            # Append to new data
            new_data.append({
                'file_path': file_path,
                'label': label,
                'spectrogram_path': spectrogram_path,
                'duration': duration,
                'prediction_confidence': "none",
                'prediction': "none",
                'prediction_correct': "none"
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Show files found by class
    print(f"\nFiles found by class:")
    for class_name in REQUIRED_CLASSES:
        count = files_by_class.get(class_name, 0)
        print(f"  {class_name}: {count} files")

    # Add new data to metadata
    new_df = pd.DataFrame(new_data)
    metadata_df = pd.concat([metadata_df, new_df], ignore_index=True)

    # Save updated metadata
    metadata_df.to_csv(METADATA_PATH, index=False)
    return metadata_df

def extract_label_from_path(file_path):
    """
    Extract class label from file path.
    Expected format: .../train/class_name/... or .../test/class_name/...
    """
    # Normalize path separators
    path_parts = file_path.replace('\\', '/').split('/')
    
    # Look for class name after 'train' or 'test'
    for i, part in enumerate(path_parts):
        if part in ['train', 'test'] and i + 1 < len(path_parts):
            potential_label = path_parts[i + 1]
            if potential_label in REQUIRED_CLASSES:
                return potential_label
    
    # If not found, check if any required class appears in the path
    for class_name in REQUIRED_CLASSES:
        if class_name in file_path:
            return class_name
    
    return None

def load_audio(file_path):
    """Load audio file and return waveform and sample rate"""
    try:
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        return waveform, SAMPLE_RATE
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def pad_or_trim(waveform, target_length=None):
    """Pad or trim waveform to target length (default: 1 second)"""
    if target_length is None:
        target_length = SAMPLE_RATE  # 1 second
    
    current_length = waveform.shape[1]
    
    if current_length > target_length:
        # Trim from center
        start = (current_length - target_length) // 2
        waveform = waveform[:, start:start + target_length]
    elif current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    return waveform

def generate_spectrogram_path(audio_path):
    """Generate spectrogram path from audio path"""
    # Replace audio directory with spectrogram directory
    rel_path = os.path.relpath(audio_path, AUDIO_DIR)
    # Change extension to .pt
    rel_path = os.path.splitext(rel_path)[0] + '.pt'
    return os.path.join(SPECTROGRAM_DIR, rel_path)

def getexperimentdata():
    """
    1. Scan through evaluation audio directory
    2. Create/update experiment metadata CSV
    """
    
    # Ensure metadata directory exists
    os.makedirs(os.path.dirname(EVALUATEDATAPATH), exist_ok=True)

    # FIXED: Added missing comma after 'highfreq_noise_count'
    column_list = [
        'file_path', 'site', 'date', 'time', 
        'alarm_count', 'non_alarm_count', 'background_count', 
        'highfreq_noise_count', 'insect_call_count', 'weather_rain_count', 'lowfreq_noise_count',
        'spectrogram_paths', 'processed'
    ]

    # Create experiment data file if empty/not exist
    if not os.path.exists(EVALUATEDATAPATH) or os.path.getsize(EVALUATEDATAPATH) == 0:
        print("Creating a new experiment file")
        experiment_data_df = pd.DataFrame(columns=column_list)
        experiment_data_df.to_csv(EVALUATEDATAPATH, index=False)
    else:
        # Load existing metadata file
        experiment_data_df = pd.read_csv(EVALUATEDATAPATH)
    
    # List out all audio files
    audio_files = []
    for root, _, files in os.walk(EVALUATEAUDIO_DIR):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} evaluation audio files in {EVALUATEAUDIO_DIR}")
    
    existing_files = set(experiment_data_df['file_path'].tolist() if 'file_path' in experiment_data_df.columns else [])
    
    # Find new files
    new_files = [f for f in audio_files if f not in existing_files]
    if len(new_files) == 0:
        print("No new evaluation audio files")
        return experiment_data_df
    else:
        print(f"Found {len(new_files)} new evaluation audio files to process")

    # Configure spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    # Process new files
    new_data = []
    for file_path in tqdm(new_files, desc="Processing evaluation audio files"):
        try:
            # Extract metadata from filename
            site, date, time = parse_filename(file_path)
            
            # Process file into 1-second segments
            _, segment_paths = process_audio_file(file_path, mel_spectrogram)
            
            if segment_paths:
                # Add entry to experiment data with all 7 class counts initialized to 0
                new_data.append({
                    'file_path': file_path,
                    'site': site,
                    'date': date,
                    'time': time,
                    'alarm_count': 0,
                    'non_alarm_count': 0,
                    'background_count': 0,
                    'highfreq_noise_count': 0,
                    'insect_call_count': 0,
                    'weather_rain_count': 0,
                    'lowfreq_noise_count': 0,
                    'spectrogram_paths': ','.join(segment_paths),
                    'processed': True  # Mark as processed since we've created the spectrograms
                })
            else:
                print(f"Warning: No segments generated for {file_path}")
                new_data.append({
                    'file_path': file_path,
                    'site': site,
                    'date': date,
                    'time': time,
                    'alarm_count': 0,
                    'non_alarm_count': 0,
                    'background_count': 0,
                    'highfreq_noise_count': 0,
                    'insect_call_count': 0,
                    'weather_rain_count': 0,
                    'lowfreq_noise_count': 0,
                    'spectrogram_paths': '',
                    'processed': False  # Mark as not processed since we couldn't generate segments
                })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Add new data to experiment data
    new_df = pd.DataFrame(new_data)
    experiment_data_df = pd.concat([experiment_data_df, new_df], ignore_index=True)

    # Save updated experiment data
    experiment_data_df.to_csv(EVALUATEDATAPATH, index=False)
    return experiment_data_df

def parse_filename(file_path):
    """
    Extract site, date, time from filename.
    Modify this function based on your filename format.
    """
    filename = os.path.basename(file_path)
    # Default values if parsing fails
    site = "unknown"
    date = "unknown"
    time = "unknown"
    
    # Example parsing - modify based on your filename format
    # Expected format: site_YYYY-MM-DD_HH-MM-SS.wav
    try:
        parts = filename.split('_')
        if len(parts) >= 3:
            site = parts[0]
            date = parts[1]
            time = parts[2].split('.')[0]  # Remove extension
    except:
        pass
    
    return site, date, time

def process_audio_file(file_path, mel_spectrogram):
    """
    Process audio file into 1-second segments and create spectrograms.
    Returns tuple of (metadata, segment_paths)
    """
    try:
        # Load audio
        waveform, sr = load_audio(file_path)
        if waveform is None:
            return None, []
        
        # Calculate segment parameters
        segment_length = SAMPLE_RATE  # 1 second
        total_length = waveform.shape[1]
        num_segments = total_length // segment_length
        
        segment_paths = []
        
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            
            # Extract segment
            segment = waveform[:, start:end]
            
            # Create spectrogram
            spec = mel_spectrogram(segment)
            spec = torch.log(spec + 1e-9)
            
            # Generate segment path
            base_path = generate_spectrogram_path(file_path)
            segment_path = base_path.replace('.pt', f'_seg{i:04d}.pt')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(segment_path), exist_ok=True)
            
            # Save segment spectrogram
            torch.save(spec, segment_path)
            segment_paths.append(segment_path)
        
        return {
            'total_segments': num_segments,
            'duration': total_length / sr
        }, segment_paths
        
    except Exception as e:
        print(f"Error processing audio file {file_path}: {e}")
        return None, []

def create_all_spectrograms(force_recreate=False):
    """
    Create spectrograms for all audio files in metadata and experiment data
    
    Args:
        force_recreate: If True, recreate spectrograms even if they exist
    """
    # First, process training data
    if os.path.exists(METADATA_PATH):
        metadata_df = pd.read_csv(METADATA_PATH)
        
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Creating training spectrograms"):
            try:
                file_path = row['file_path']
                spectrogram_path = row['spectrogram_path']
                
                # Skip if spectrogram exists and we're not forcing recreation
                if os.path.exists(spectrogram_path) and not force_recreate:
                    continue
                
                # Load and process audio
                waveform, sr = load_audio(file_path)
                if waveform is None:
                    print(f"Skipping {file_path} - could not load audio")
                    continue
                
                # Pad or trim to 1 second
                waveform = pad_or_trim(waveform)
                
                # Create spectrogram
                spec = mel_spectrogram(waveform)
                spec = torch.log(spec + 1e-9)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(spectrogram_path), exist_ok=True)
                
                # Save spectrogram
                torch.save(spec, spectrogram_path)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    else:
        print("Training metadata file not found. Run getmetadata first.")
        
    # Next, process evaluation data
    if os.path.exists(EVALUATEDATAPATH) and os.path.getsize(EVALUATEDATAPATH) > 0:
        try:
            experiment_df = pd.read_csv(EVALUATEDATAPATH)
            
            # Find unprocessed files
            unprocessed_files = experiment_df[experiment_df['processed'] == False]
            
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            )
            
            for idx, row in tqdm(unprocessed_files.iterrows(), total=len(unprocessed_files), desc="Creating evaluation spectrograms"):
                try:
                    file_path = row['file_path']
                    
                    # Process file into segments and create spectrograms
                    _, segment_paths = process_audio_file(file_path, mel_spectrogram)
                    
                    if segment_paths:
                        # Update spectrogram paths
                        experiment_df.at[idx, 'spectrogram_paths'] = ','.join(segment_paths)
                        experiment_df.at[idx, 'processed'] = True
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            
            # Save updated experiment data
            experiment_df.to_csv(EVALUATEDATAPATH, index=False)
        except pd.errors.EmptyDataError:
            print(f"Warning: {EVALUATEDATAPATH} was empty. Creating new DataFrame.")
            column_list = [
                'file_path', 'site', 'date', 'time', 
                'alarm_count', 'non_alarm_count', 'background_count', 
                'highfreq_noise_count', 'insect_call_count', 'weather_rain_count', 'lowfreq_noise_count',
                'spectrogram_paths', 'processed'
            ]
            experiment_df = pd.DataFrame(columns=column_list)
            experiment_df.to_csv(EVALUATEDATAPATH, index=False)
    else:
        print("Evaluation metadata file not found or empty. Run getexperimentdata first.")

def check_class_distribution(metadata_df):
    """Check the distribution of classes across all files in metadata"""
    if 'label' not in metadata_df.columns:
        return {"error": "No label column in metadata"}
    
    # Count occurrences of each class
    class_counts = metadata_df['label'].value_counts().to_dict()
    
    # Fill in zero counts for any missing classes
    for cls in REQUIRED_CLASSES:
        if cls not in class_counts:
            class_counts[cls] = 0
    
    total_samples = sum(class_counts.values())
    
    distribution = {
        "total_samples": total_samples,
        "class_counts": class_counts,
        "class_percentages": {cls: (count/total_samples*100) if total_samples > 0 else 0 
                             for cls, count in class_counts.items()}
    }
    
    return distribution

def verify_few_shot_requirements(metadata_df, n_way=N_WAY, k_shot=N_SUPPORT, query_size=N_QUERY, test_size=TEST_SIZE):
    """
    Verify the latest dataset meets few shot requirements to prevent future errors
    
    Checks:
    1. Total samples per class
    2. Sufficient samples for support set from train directory
    3. Sufficient samples for query set from train directory
    4. Sufficient samples for test set from test directory
    """
    if 'label' not in metadata_df.columns or 'file_path' not in metadata_df.columns:
        return {
            "meets_requirements": False,
            "error": "Missing label or file_path column in metadata",
            "suggestion": "Run preprocessing first to generate metadata with labels"
        }
    
    # If no data at all
    if len(metadata_df) == 0:
        return {
            "meets_requirements": False,
            "error": "No data found in metadata",
            "suggestion": f"Add audio files to {AUDIO_DIR} in the expected directory structure"
        }
    
    # Total samples needed per class
    total_samples_needed = k_shot + query_size + test_size
    
    # Detailed verification results
    verification_results = {
        "meets_requirements": True,
        "class_details": {}
    }
    
    for cls in REQUIRED_CLASSES:
        # Separate train and test samples
        train_samples = metadata_df[
            (metadata_df['label'] == cls) & 
            (metadata_df['file_path'].str.contains('train/'))
        ]
        test_samples = metadata_df[
            (metadata_df['label'] == cls) & 
            (metadata_df['file_path'].str.contains('test/'))
        ]
        
        # Verify support set samples from train directory
        support_samples = train_samples.head(k_shot)
        if len(support_samples) < k_shot:
            verification_results["meets_requirements"] = False
            verification_results["class_details"][cls] = {
                "train_samples": len(train_samples),
                "support_samples": len(support_samples),
                "support_samples_needed": k_shot,
                "error": f"Insufficient train support samples. Need {k_shot}, have {len(support_samples)}"
            }
            continue
        
        # Verify query set samples from train directory
        query_samples = train_samples.iloc[k_shot:k_shot+query_size]
        if len(query_samples) < query_size:
            verification_results["meets_requirements"] = False
            verification_results["class_details"][cls] = {
                "train_samples": len(train_samples),
                "query_samples": len(query_samples),
                "query_samples_needed": query_size,
                "error": f"Insufficient train query samples. Need {query_size}, have {len(query_samples)}"
            }
            continue
        
        # Verify test set samples from test directory (optional)
        if len(test_samples) > 0:
            test_samples_subset = test_samples.head(test_size)
            if len(test_samples_subset) < test_size:
                verification_results["meets_requirements"] = False
                verification_results["class_details"][cls] = {
                    "test_samples": len(test_samples),
                    "test_samples_subset": len(test_samples_subset),
                    "test_samples_needed": test_size,
                    "error": f"Insufficient test samples. Need {test_size}, have {len(test_samples_subset)}"
                }
                continue
        
        # If we've made it this far, this class passes
        verification_results["class_details"][cls] = {
            "train_total_samples": len(train_samples),
            "test_total_samples": len(test_samples),
            "support_samples": len(support_samples),
            "query_samples": len(query_samples),
            "test_samples": len(test_samples) if len(test_samples) > 0 else 0,
            "status": "PASS"
        }
    
    # If any class failed, provide a suggestion
    if not verification_results["meets_requirements"]:
        verification_results["suggestion"] = (
            f"Need {k_shot} support samples, {query_size} query samples from train directories, "
            f"and {test_size} test samples from test directories for all classes: "
            f"{', '.join(REQUIRED_CLASSES)}. "
            "Check the class_details for specific requirements."
        )
    
    return verification_results