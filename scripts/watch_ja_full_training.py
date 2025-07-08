#!/usr/bin/env python3
"""
ja-full学習の進捗を監視するスクリプト
"""

import os
import time
import subprocess
from datetime import datetime
import re

def get_training_progress():
    """ログファイルから最新の進捗を取得"""
    log_file = "outputs/train_ja_full.log"
    
    if not os.path.exists(log_file):
        return None
    
    # 最新の進捗情報を取得
    try:
        with subprocess.Popen(['tail', '-n', '100', log_file], stdout=subprocess.PIPE, text=True) as proc:
            output = proc.stdout.read()
        
        # ステップ数とロスを抽出
        step_pattern = r'Epoch 1:\s+(\d+)%.*?(\d+)/(\d+)'
        loss_pattern = r'Step (\d+), Loss: ([\d.]+)'
        
        steps = []
        losses = []
        
        for line in output.split('\n'):
            step_match = re.search(step_pattern, line)
            if step_match:
                current_step = int(step_match.group(2))
                total_steps = int(step_match.group(3))
                progress_pct = int(step_match.group(1))
                
            loss_match = re.search(loss_pattern, line)
            if loss_match:
                losses.append((int(loss_match.group(1)), float(loss_match.group(2))))
        
        # 最新の情報を返す
        if 'current_step' in locals():
            return {
                'current_step': current_step,
                'total_steps': total_steps,
                'progress_pct': progress_pct,
                'losses': losses
            }
    except Exception as e:
        print(f"エラー: {e}")
    
    return None

def check_process_status():
    """プロセスが実行中かチェック"""
    try:
        result = subprocess.run(['pgrep', '-f', 'train_ja_full.py'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def main():
    print("🔍 ja-full 学習監視を開始します...")
    print("=" * 50)
    
    last_step = 0
    start_time = time.time()
    
    while True:
        # プロセスチェック
        if not check_process_status():
            print("\n✅ 学習プロセスが終了しました！")
            break
        
        # 進捗取得
        progress = get_training_progress()
        
        if progress:
            current_step = progress['current_step']
            total_steps = progress['total_steps']
            progress_pct = progress['progress_pct']
            
            # 進捗が更新された場合のみ表示
            if current_step > last_step:
                elapsed = time.time() - start_time
                steps_per_sec = (current_step - last_step) / 30 if last_step > 0 else 0
                
                if steps_per_sec > 0:
                    remaining_steps = total_steps - current_step
                    eta_seconds = remaining_steps / steps_per_sec
                    eta_minutes = int(eta_seconds / 60)
                else:
                    eta_minutes = "計算中"
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
                print(f"📊 進捗: {current_step:,}/{total_steps:,} ({progress_pct}%)")
                print(f"⚡ 速度: {steps_per_sec:.1f} steps/s")
                print(f"⏱️  推定残り時間: {eta_minutes} 分")
                
                # 最新のロスを表示
                if progress['losses']:
                    latest_loss = progress['losses'][-1]
                    print(f"📉 最新ロス: Step {latest_loss[0]}, Loss: {latest_loss[1]:.4f}")
                
                last_step = current_step
        
        # 30秒待機
        time.sleep(30)
    
    # 学習完了時の処理
    final_model = "./outputs/provence-ja-full/final-model"
    if os.path.exists(final_model):
        print(f"\n🎉 最終モデル保存完了: {final_model}")
        print("\n次のコマンドで評価を実行できます:")
        print("uv run python scripts/evaluate_ja_full.py")

if __name__ == "__main__":
    main()