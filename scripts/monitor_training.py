#!/usr/bin/env python3
"""
学習進行状況モニタリングスクリプト
"""

import os
import time
import subprocess
from datetime import datetime

def check_training_status():
    """学習の進行状況をチェック"""
    print(f"\n=== 学習状況確認 ({datetime.now().strftime('%H:%M:%S')}) ===")
    
    # プロセス確認
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = [line for line in result.stdout.split('\n') if 'train_provence' in line and 'full' in line and 'grep' not in line]
        
        if lines:
            print("🟢 学習プロセス実行中:")
            for line in lines:
                parts = line.split()
                cpu = parts[2]
                memory = parts[3]
                time_used = parts[9]
                print(f"   CPU: {cpu}%, Memory: {memory}%, Time: {time_used}")
        else:
            print("🔴 学習プロセスが見つかりません")
            return False
    except Exception as e:
        print(f"❌ プロセス確認エラー: {e}")
    
    # ログファイル確認
    log_file = "logs/train_full_corrected.log"
    if os.path.exists(log_file):
        print("\n📋 最新ログ (最後の10行):")
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(f"   {line.strip()}")
        except Exception as e:
            print(f"❌ ログ読み込みエラー: {e}")
    else:
        print("📋 ログファイルが見つかりません")
    
    # モデルファイル確認
    final_model_path = "outputs/provence-ja-full/final-model"
    if os.path.exists(final_model_path):
        stat = os.stat(final_model_path)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        print(f"\n📁 Final model 最終更新: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # チェックポイント確認
    checkpoint_dir = "outputs/provence-ja-full"
    if os.path.exists(checkpoint_dir):
        try:
            files = os.listdir(checkpoint_dir)
            checkpoints = [f for f in files if 'checkpoint-' in f and 'best' in f]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
                stat = os.stat(os.path.join(checkpoint_dir, latest_checkpoint))
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                print(f"💾 最新チェックポイント: {latest_checkpoint} ({mod_time.strftime('%H:%M:%S')})")
        except Exception as e:
            print(f"❌ チェックポイント確認エラー: {e}")
    
    return True

def main():
    """5分ごとに学習状況をモニタリング"""
    print("🔍 学習モニタリング開始 (5分間隔)")
    print("Ctrl+C で停止")
    
    try:
        while True:
            running = check_training_status()
            if not running:
                print("\n✅ 学習が完了したようです")
                break
            
            print(f"\n⏰ 次回チェック: {(datetime.now()).strftime('%H:%M:%S')} + 5分")
            time.sleep(300)  # 5分間隔
            
    except KeyboardInterrupt:
        print("\n🛑 モニタリング停止")

if __name__ == "__main__":
    main()