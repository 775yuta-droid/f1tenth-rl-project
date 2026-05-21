"""
F1TENTH インタラクティブ・スポーンエディタ (Web版)
"""

import sys
import os
import yaml
import json
import base64
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
from io import BytesIO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src import config

PORT = 8988

def get_map_data():
    map_path_raw = config.MAP_PATH
    map_yaml = map_path_raw + ".yaml"
    if not os.path.exists(map_yaml):
        if map_path_raw.startswith('/workspace/'):
            rel_path = map_path_raw.replace('/workspace/', '')
            map_yaml = os.path.join(PROJECT_ROOT, rel_path + ".yaml")
        else:
            map_yaml = os.path.join(PROJECT_ROOT, "my_maps", "testmap-tamoku", "map-tamoku.yaml")
    with open(map_yaml, 'r') as f:
        map_conf = yaml.safe_load(f)
    origin = map_conf['origin']
    resolution = map_conf['resolution']
    img_name = map_conf['image']
    img_path = os.path.join(os.path.dirname(map_yaml), img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(PROJECT_ROOT, "my_maps", "testmap-tamoku", img_name)
    img = Image.open(img_path).convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return {
        "img_base64": img_base64,
        "width": img.width,
        "height": img.height,
        "origin": origin,
        "resolution": resolution,
        "car_width": config.CAR_WIDTH,
        "car_length": config.CAR_LENGTH,
        "poses": [list(p) for p in config.START_POSES] if hasattr(config, 'START_POSES') else [list(config.START_POSE)]
    }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>F1TENTH Spawn Editor</title>
    <style>
        body { background: #1a1a1a; color: #eee; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
               margin: 0; display: flex; flex-direction: column; align-items: center; height: 100vh; overflow: hidden; }
        .toolbar { width: 100%; padding: 15px; background: #2d2d2d; box-shadow: 0 2px 10px rgba(0,0,0,0.5); 
                    display: flex; gap: 20px; align-items: center; justify-content: center; z-index: 10; }
        .canvas-container { flex: 1; display: flex; align-items: center; justify-content: center; width: 100%; overflow: hidden; padding: 20px; position: relative; }
        canvas { background: #000; box-shadow: 0 0 20px rgba(0,0,0,1); cursor: crosshair; image-rendering: pixelated; }
        button { padding: 10px 20px; border-radius: 4px; border: none; font-weight: bold; cursor: pointer; transition: 0.2s; }
        #save-btn { background: #2ecc71; color: white; }
        #save-btn:hover { background: #27ae60; }
        #save-btn:disabled { background: #7f8c8d; cursor: not-allowed; }
        .hint { font-size: 0.9em; color: #aaa; }
        kbd { background: #444; padding: 2px 5px; border-radius: 3px; border: 1px solid #666; font-family: monospace; }
        .status { position: fixed; bottom: 20px; right: 20px; padding: 10px 20px; border-radius: 4px; display: none; z-index: 100; }
    </style>
</head>
<body>
    <div class="toolbar">
        <div style="font-weight: bold; font-size: 1.2em;">F1TENTH Spawn Editor</div>
        <button id="save-btn">CONFIG.PY に保存</button>
        <div class="hint">
            <kbd>左クリック</kbd> 追加/選択 | <kbd>ドラッグ</kbd> 移動 | <kbd>Shift+ドラッグ</kbd> 回転 | <kbd>右クリック</kbd> 削除 | <kbd>ホイール</kbd> ズーム | <kbd>中クリック/Ctrl+ドラッグ</kbd> パン
        </div>
    </div>
    <div class="canvas-container" id="container">
        <canvas id="mapCanvas"></canvas>
    </div>
    <div id="status-msg" class="status"></div>

    <script>
        const data = {{DATA_JSON}};
        const canvas = document.getElementById('mapCanvas');
        const ctx = canvas.getContext('2d');
        const mapImg = new Image();
        
        let poses = data.poses.map(p => ({x: p[0], y: p[1], yaw: p[2]}));
        let selectedIdx = null;
        let isDragging = false;
        let isRotating = false;
        let isPanning = false;
        
        let zoom = 1.0;
        let panX = 0;
        let panY = 0;

        mapImg.onload = () => {
            canvas.width = window.innerWidth * 0.9;
            canvas.height = window.innerHeight * 0.8;
            // Center the small map initially
            panX = (canvas.width - data.width * zoom) / 2;
            panY = (canvas.height - data.height * zoom) / 2;
            draw();
        };
        mapImg.src = "data:image/png;base64," + data.img_base64;

        function worldToPixel(x, y) {
            const px = (x - data.origin[0]) / data.resolution;
            const py = data.height - (y - data.origin[1]) / data.resolution;
            return {x: px, y: py};
        }

        function pixelToWorld(px, py) {
            const x = px * data.resolution + data.origin[0];
            const y = (data.height - py) * data.resolution + data.origin[1];
            return {x, y};
        }

        function screenToMap(sx, sy) {
            return {
                x: (sx - panX) / zoom,
                y: (sy - panY) / zoom
            };
        }

        function draw() {
            ctx.fillStyle = '#111';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.save();
            ctx.translate(panX, panY);
            ctx.scale(zoom, zoom);
            
            // Draw Map
            ctx.drawImage(mapImg, 0, 0);
            
            // Draw Poses
            poses.forEach((p, i) => {
                const pos = worldToPixel(p.x, p.y);
                const isSelected = i === selectedIdx;
                
                ctx.save();
                ctx.translate(pos.x, pos.y);
                ctx.rotate(-p.yaw);
                
                const w = data.car_length / data.resolution;
                const h = data.car_width / data.resolution;
                
                ctx.fillStyle = isSelected ? 'rgba(0, 255, 255, 0.6)' : 'rgba(0, 255, 0, 0.6)';
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1 / zoom; // Keep lines thin even when zoomed
                ctx.fillRect(-w/2, -h/2, w, h);
                ctx.strokeRect(-w/2, -h/2, w, h);
                
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.lineTo(w/2 + 5, 0);
                ctx.stroke();
                
                ctx.restore();
                
                // Label (Red as requested)
                ctx.fillStyle = '#ff3333';
                ctx.font = `bold ${14/zoom}px Arial`;
                ctx.fillText("#" + i, pos.x + 8/zoom, pos.y - 8/zoom);
            });
            
            ctx.restore();
        }

        canvas.onmousedown = (e) => {
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            
            if (e.button === 1 || (e.button === 0 && e.ctrlKey)) { // Middle click or Ctrl+Left click to pan
                isPanning = true;
                return;
            }

            const mapPos = screenToMap(mx, my);
            
            let found = -1;
            poses.forEach((p, i) => {
                const pos = worldToPixel(p.x, p.y);
                const dist = Math.sqrt((pos.x - mapPos.x)**2 + (pos.y - mapPos.y)**2);
                if (dist < 15 / zoom) found = i;
            });

            if (e.button === 2) { // Right click
                if (found !== -1) {
                    poses.splice(found, 1);
                    selectedIdx = null;
                }
                e.preventDefault();
            } else {
                if (found !== -1) {
                    selectedIdx = found;
                    isDragging = true;
                    isRotating = e.shiftKey;
                } else {
                    const world = pixelToWorld(mapPos.x, mapPos.y);
                    poses.push({x: world.x, y: world.y, yaw: 0});
                    selectedIdx = poses.length - 1;
                    isDragging = true;
                }
            }
            draw();
        };

        window.onmousemove = (e) => {
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;

            if (isPanning) {
                panX += e.movementX;
                panY += e.movementY;
                draw();
                return;
            }

            if (!isDragging || selectedIdx === null) return;
            
            const mapPos = screenToMap(mx, my);
            
            if (isRotating) {
                const pos = worldToPixel(poses[selectedIdx].x, poses[selectedIdx].y);
                const angle = Math.atan2(pos.y - mapPos.y, mapPos.x - pos.x);
                poses[selectedIdx].yaw = angle;
            } else {
                const world = pixelToWorld(mapPos.x, mapPos.y);
                poses[selectedIdx].x = world.x;
                poses[selectedIdx].y = world.y;
            }
            draw();
        };

        window.onmouseup = () => {
            isDragging = false;
            isRotating = false;
            isPanning = false;
        };

        canvas.onwheel = (e) => {
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;

            const mapPosPre = screenToMap(mx, my);
            
            const zoomSpeed = 0.1;
            if (e.deltaY < 0) zoom *= (1 + zoomSpeed);
            else zoom /= (1 + zoomSpeed);
            
            zoom = Math.max(0.1, Math.min(zoom, 50));

            // Adjust pan to zoom around mouse position
            panX = mx - mapPosPre.x * zoom;
            panY = my - mapPosPre.y * zoom;

            draw();
            e.preventDefault();
        };

        canvas.oncontextmenu = (e) => e.preventDefault();

        document.getElementById('save-btn').onclick = async () => {
            const btn = document.getElementById('save-btn');
            btn.disabled = true;
            btn.innerText = "保存中...";
            try {
                const resp = await fetch('/save', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(poses.map(p => [p.x, p.y, p.yaw]))
                });
                const res = await resp.json();
                showStatus(res.msg, res.success ? '#27ae60' : '#c0392b');
            } catch (e) {
                showStatus("Error: " + e, '#c0392b');
            }
            btn.disabled = false;
            btn.innerText = "CONFIG.PY に保存";
        };

        function showStatus(msg, color) {
            const el = document.getElementById('status-msg');
            el.innerText = msg;
            el.style.background = color;
            el.style.display = 'block';
            setTimeout(() => { el.style.display = 'none'; }, 3000);
        }

        window.onresize = () => {
            canvas.width = window.innerWidth * 0.9;
            canvas.height = window.innerHeight * 0.8;
            draw();
        };
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            data = get_map_data()
            html = HTML_TEMPLATE.replace('{{DATA_JSON}}', json.dumps(data))
            self.wfile.write(html.encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/save':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            poses = json.loads(post_data)
            success, msg = self.save_to_config(poses)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"success": success, "msg": msg}).encode())

    def save_to_config(self, poses):
        config_path = os.path.join(PROJECT_ROOT, 'src', 'config.py')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            pattern = r'(START_POSES\s*=\s*\[)(.*?)(\n\s*\])'
            new_list_str = "\n"
            for i, p in enumerate(poses):
                new_list_str += f"    [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}],  # Pose {i}\n"
            if re.search(pattern, content, re.DOTALL):
                new_content = re.sub(pattern, r'\1' + new_list_str + r'\3', content, flags=re.DOTALL)
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True, f"{len(poses)} 件のスポーン地点を保存しました！"
            else:
                return False, "config.py 内に START_POSES リストが見つかりませんでした。"
        except Exception as e:
            return False, f"保存エラー: {str(e)}"

    def log_message(self, format, *args):
        return

def run():
    print("\n" + "="*60)
    print("F1TENTH Interactive Spawn Editor (V2 Web with Zoom)")
    print(f"ブラウザで以下にアクセスしてください:")
    print(f"    http://localhost:{PORT}/")
    print("="*60 + "\n")
    server = HTTPServer(('0.0.0.0', PORT), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nサーバーを停止しました。")
        server.server_close()

if __name__ == '__main__':
    run()
