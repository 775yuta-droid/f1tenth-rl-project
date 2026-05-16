import zipfile
import json
import os

def detect_algo(model_path):
    """
    モデルのzipファイル内のメタデータを読み取り、アルゴリズム名を返す。
    
    Args:
        model_path (str): モデルファイルのパス(拡張子なしでも可)
        
    Returns:
        str: 'ppo', 'sac', 'td3' のいずれか。判別不能な場合は None。
    """
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    
    if not os.path.exists(model_path):
        return None
    
    try:
        # まずはファイル名から推測
        filename_lower = os.path.basename(model_path).lower()
        if "ppo" in filename_lower:
            return "ppo"
        elif "sac" in filename_lower:
            return "sac"
        elif "td3" in filename_lower:
            return "td3"

        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            if 'data' not in zip_ref.namelist():
                return None
            
            with zip_ref.open('data') as data_file:
                content = data_file.read().decode('utf-8')
                data = json.loads(content)
                
                # Check for class name in data
                # SB3 saves the class info in several possible places in the 'data' JSON
                class_path = ""
                if "policy_class" in data and ":serialized:" in data["policy_class"]:
                    # Serialized data is harder to parse, but usually __module__ is present
                    class_path = data["policy_class"].get("__module__", "")
                
                if not class_path:
                    class_path = data.get("__module__", "")
                
                # If not found in module, check if it's explicitly in some other field
                # or just use the content of the data file as a string search as a fallback
                class_path_lower = class_path.lower()
                
                if "ppo" in class_path_lower:
                    return "ppo"
                elif "sac" in class_path_lower:
                    return "sac"
                elif "td3" in class_path_lower:
                    return "td3"
                
                # Fallback: search the entire content for algorithm keywords if module path is missing
                if "stable_baselines3.ppo" in content:
                    return "ppo"
                elif "stable_baselines3.sac" in content:
                    return "sac"
                elif "stable_baselines3.td3" in content:
                    return "td3"
                    
    except Exception as e:
        print(f"[ALGO] Detection error for {model_path}: {e}")
    
    return None

def get_algo_class(algo_name):
    """
    アルゴリズム名に対応するStable Baselines3のクラスを返す。
    
    Args:
        algo_name (str): 'ppo', 'sac', 'td3'
        
    Returns:
        Type: SB3 Algorithm class
    """
    if not algo_name:
        return None
        
    algo_name = algo_name.lower()
    if algo_name == "ppo":
        from stable_baselines3 import PPO
        return PPO
    elif algo_name == "sac":
        from stable_baselines3 import SAC
        return SAC
    elif algo_name == "td3":
        from stable_baselines3 import TD3
        return TD3
    else:
        return None
