import numpy as np
import cv2
import os
from pathlib import Path
import configparser
from .pylidc_config import configure_pylidc,import_pylidc
from .config import DataPrepConfig
configparser.SafeConfigParser = configparser.ConfigParser
np.int = np.int64
np.float = np.float64
np.bool = np.bool_
np.object = np.object_
np.str = np.str_


DEFAULT_CONFIG = {
    'data_path': r"E:\FinalsProject\Datasets\CancerDetection\images\manifest-1600709154662\LIDC-IDRI",
    'output_dir': r".\DetectionModel\datasets",
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'min_diameter': 3.0,
    'max_diameter': 100.0,
    'slices_per_nodule': 3,
    'seed': 42,
    'debug': False,
    'log_freq': 5
}


config = DataPrepConfig(
        data_path=DEFAULT_CONFIG['data_path'],
        output_dir=DEFAULT_CONFIG['output_dir'],
        train_ratio=DEFAULT_CONFIG['train_ratio'],
        val_ratio=DEFAULT_CONFIG['val_ratio'],
        test_ratio=DEFAULT_CONFIG['test_ratio'],
        min_nodule_diameter=DEFAULT_CONFIG['min_diameter'],
        max_nodule_diameter=DEFAULT_CONFIG['max_diameter'],
        slices_per_nodule=DEFAULT_CONFIG['slices_per_nodule'],
        random_seed=DEFAULT_CONFIG['seed'],
        log_freq=DEFAULT_CONFIG['log_freq']
    )


configure_pylidc(config.data_path)

pl=import_pylidc()

# --- הגדרת התיקון (אותה פונקציה שנתתי לך קודם) ---
def robust_windowing(volume, center=-600, width=1500):
    # הגנה מפני NaN
    if np.isnan(volume).any():
        print("  [!] Warning: NaN values found! Fixing...")
        volume = np.nan_to_num(volume, nan=-1000.0)

    # בדיקת טווחים (Offset Check)
    vol_min, vol_max = volume.min(), volume.max()
    
    # אם המינימום גבוה מדי (למשל 0 במקום -1000), כנראה שיש Offset
    if vol_min >= 0:
        print(f"  [!] Detected Offset issue (Min: {vol_min}). Trying to fix...")
        volume = volume - 1024  # תיקון נפוץ ל-CT
        if volume.min() < -2000: # אם תיקנו יותר מדי
             volume = volume + 1024 # בטל תיקון

    # ביצוע ה-Windowing
    lower = center - (width / 2.0)
    upper = center + (width / 2.0)
    
    # Clip ו-Normalize
    windowed = np.clip(volume, lower, upper)
    windowed = (windowed - lower) / width
    
    # *** התיקון הקריטי למניעת שלג/רעש ***
    windowed = np.clip(windowed, 0.0, 1.0)
    
    return (windowed * 255.0).astype(np.uint8)

def main():
    # הגדרות
    OUTPUT_DIR = "debug_preview"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Querying LIDC database...")
    scans = pl.query(pl.Scan).all()
    
    print(f"Found {len(scans)} scans. Starting rapid diagnosis...")
    print(f"Saving previews to: {os.path.abspath(OUTPUT_DIR)}")
    print("-" * 50)

    # נבדוק 20 סריקות אקראיות (או את כולן אם אתה רוצה, אבל זה ייקח זמן)
    # כדי למצוא בעיות מהר, נרוץ על קפיצות
    problematic_found = 0
    
    # רוץ על כל הסריקות בקפיצות של 10 (כדי לקבל מדגם מייצג מהר)
    for i in range(0, len(scans), 10): 
        scan = scans[i]
        pid = scan.patient_id
        
        try:
            print(f"Checking {pid}...", end="", flush=True)
            
            # טעינת הנפח (DICOM)
            vol = scan.to_volume()
            
            # בדיקת סטטיסטיקה גולמית
            v_min, v_max, v_mean = vol.min(), vol.max(), vol.mean()
            
            # לוגיקה לזיהוי "סריקה בעייתית" פוטנציאלית
            is_suspicious = False
            if v_min > -500: # ב-CT תקין תמיד יש אוויר (-1000)
                print(f" SUSPICIOUS! (High Min: {v_min})", end="")
                is_suspicious = True
            elif v_max < 500: # ב-CT תקין תמיד יש עצם (+400 ומעלה)
                print(f" SUSPICIOUS! (Low Max: {v_max})", end="")
                is_suspicious = True
            else:
                print(" OK.", end="")

            # אנחנו נשמור תמונה אם היא חשודה, או סתם אחת ל-5 תקינות כדי לוודא
            if is_suspicious or (i % 50 == 0):
                # הפעלת התיקון
                processed_vol = robust_windowing(vol)
                
                # שליפת סלייס אמצעי
                mid_slice = processed_vol.shape[0] // 2
                img = processed_vol[mid_slice]
                
                # שמירה
                status = "BAD" if is_suspicious else "OK"
                fname = f"{OUTPUT_DIR}/{status}_{pid}_slice{mid_slice}.jpg"
                cv2.imwrite(fname, img)
                
                if is_suspicious:
                    problematic_found += 1
            
            print() # ירידת שורה

        except Exception as e:
            print(f" Error: {e}")

        # אם מצאנו 5 בעייתיות ותיקנו אותן - אפשר לעצור ולבדוק
        if problematic_found >= 5:
            print("\nFound enough suspicious scans for testing. Stopping.")
            break

    print("-" * 50)
    print("Done. Check the 'debug_preview' folder now.")

if __name__ == "__main__":
    main()