#!/bin/bash
# Quick Demo Startup Script - Fledgling Agent Parity Platform
# Shows SLM vs LLM performance using real evaluation data

set -e

echo "🚀 Starting Fledgling Demo..."
echo ""

# Check if already running
if curl -s http://localhost:4000/health > /dev/null 2>&1; then
    echo "✅ Backend already running on http://localhost:4000"
else
    echo "📦 Starting backend..."
    cd backend
    pkill -f "ts-node-dev" 2>/dev/null || true
    nohup pnpm dev > /tmp/fledgling-backend.log 2>&1 &
    cd ..
    echo "   Waiting for backend to start..."
    for i in {1..15}; do
        if curl -s http://localhost:4000/health > /dev/null 2>&1; then
            echo "   ✅ Backend started"
            break
        fi
        sleep 1
        echo -n "."
    done
fi

# Check frontend
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "✅ Frontend already running on http://localhost:5173"
else
    echo "🎨 Starting frontend..."
    cd frontend
    nohup pnpm dev > /tmp/fledgling-frontend.log 2>&1 &
    cd ..
    echo "   ✅ Frontend starting (will be ready in ~10 seconds)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Fledgling Demo Ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Open in browser:"
echo "   → http://localhost:5173"
echo ""
echo "🧭 Available pages:"
echo "   • Ops Dashboard:   http://localhost:5173/"
echo "   • Trace Console:   http://localhost:5173/traces"
echo "   • Metrics:         http://localhost:5173/metrics"
echo ""
echo "🔧 API Endpoints:"
echo "   • Metrics:         http://localhost:4000/api/metrics/comparison"
echo "   • Traces:          http://localhost:4000/api/traces"
echo "   • Training Status: http://localhost:4000/api/training/status"
echo ""
echo "📈 What to show in demo:"
echo "   1. Trace Console → Shows 10 real agent traces from datasets"
echo "   2. Metrics Page  → SLM (88% valid, 32% F1) vs Azure (100%, 60% F1)"
echo "   3. Training      → Completed run (114 steps, 4m 51s)"
echo "   4. Ops Dashboard → Model selector, HF upload, token management"
echo ""
echo "🎬 Demo Flow:"
echo "   'We capture agent traces → Export to datasets → Fine-tune SLM → Compare'"
echo "   'Current structured adapter: 53% parity with Azure LLM'"
echo "   'Cost: \$30/1M tokens (Azure) → \$0.10/1M tokens (local SLM)'"
echo ""
echo "🛑 To stop: pkill -f 'pnpm dev' or Ctrl+C in terminal"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Quick health check
echo "🏥 Quick health check..."
sleep 2
if curl -s http://localhost:4000/api/traces 2>&1 | grep -q "samples"; then
    echo "✅ Backend serving traces data"
else
    echo "⚠️  Backend may still be starting - wait 10s and refresh browser"
fi

echo ""
echo "✨ Demo ready! Open http://localhost:5173 in your browser"
