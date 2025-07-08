#!/bin/bash
# Monitor ja-full training progress

echo "=== ja-full 学習監視 ==="
echo "進捗確認用スクリプト"
echo ""

while true; do
    # Check if process is still running
    if ! ps aux | grep -q "[t]rain_ja_full.py"; then
        echo "🏁 学習プロセスが終了しました"
        break
    fi
    
    # Get last 20 lines of log
    echo "📊 現在の進捗:"
    tail -n 20 outputs/train_ja_full.log | grep -E "(Epoch|Loss:|✅|💾)"
    
    # Check for checkpoints
    echo ""
    echo "💾 保存されたチェックポイント:"
    ls -lht outputs/provence-ja-full/checkpoint-* 2>/dev/null | head -5
    
    # Wait 30 seconds before next check
    echo ""
    echo "次の確認まで30秒待機... (Ctrl+Cで終了)"
    sleep 30
    clear
done

echo "✅ 監視終了"