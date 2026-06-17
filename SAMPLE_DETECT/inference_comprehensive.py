"""
Comprehensive YOLO Inference Script for Parks Surveillance
==========================================================

Advanced inference script with object tracking, zone detection, trajectory
visualization, and analytics capabilities.

Features:
- Object tracking across frames (ByteTrack)
- Zone-based detection and alerts
- Trajectory visualization
- Stationary object detection
- Real-time analytics
- Multiple output formats (video, JSON, CSV)

Usage Examples:
    # Basic tracking
    python inference_comprehensive.py --source video.mp4 --track
    
    # Zone detection
    python inference_comprehensive.py --source video.mp4 --zones "[[100,100,500,100,500,400,100,400]]"
    
    # Full analytics with trajectory
    python inference_comprehensive.py --source video.mp4 --track --save-trajectory --analytics
    
    # Export to JSON
    python inference_comprehensive.py --source video.mp4 --track --export-json
"""

from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path
import torch
import json
import csv
from collections import defaultdict, deque
import numpy as np
from datetime import datetime


class ObjectTracker:
    """Track object statistics and trajectories."""
    
    def __init__(self, max_history=50):
        self.tracks = defaultdict(lambda: {
            'positions': deque(maxlen=max_history),
            'class_name': None,
            'first_seen': None,
            'last_seen': None,
            'stationary_frames': 0
        })
        self.max_history = max_history
    
    def update(self, track_id, bbox, class_name, frame_num):
        """Update track information."""
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        track = self.tracks[track_id]
        track['positions'].append(center)
        track['class_name'] = class_name
        track['last_seen'] = frame_num
        
        if track['first_seen'] is None:
            track['first_seen'] = frame_num
        
        # Check if stationary
        if len(track['positions']) >= 10:
            recent_positions = list(track['positions'])[-10:]
            distances = [np.sqrt((recent_positions[i][0] - recent_positions[i-1][0])**2 + 
                                (recent_positions[i][1] - recent_positions[i-1][1])**2)
                        for i in range(1, len(recent_positions))]
            avg_distance = np.mean(distances)
            
            if avg_distance < 5:  # Threshold for stationary
                track['stationary_frames'] += 1
            else:
                track['stationary_frames'] = 0
    
    def is_stationary(self, track_id, threshold=30):
        """Check if object is stationary."""
        return self.tracks[track_id]['stationary_frames'] > threshold
    
    def get_trajectory(self, track_id):
        """Get trajectory points for a track."""
        return list(self.tracks[track_id]['positions'])
    
    def get_all_tracks(self):
        """Get all track data."""
        return dict(self.tracks)


class ZoneDetector:
    """Detect objects in defined zones."""
    
    def __init__(self, zones):
        """
        Initialize with zones.
        
        Args:
            zones: List of polygons, each polygon is list of (x,y) points
        """
        self.zones = zones
        self.zone_counts = defaultdict(int)
        self.zone_history = defaultdict(list)
    
    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting."""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def check_zones(self, bbox, frame_num):
        """Check which zones the object is in."""
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        zones_detected = []
        
        for i, zone in enumerate(self.zones):
            if self.point_in_polygon(center, zone):
                zones_detected.append(i)
                self.zone_counts[i] += 1
                self.zone_history[i].append(frame_num)
        
        return zones_detected
    
    def draw_zones(self, frame):
        """Draw zones on frame."""
        overlay = frame.copy()
        for i, zone in enumerate(self.zones):
            pts = np.array(zone, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            
            # Add zone label
            centroid = np.mean(zone, axis=0).astype(int)
            cv2.putText(overlay, f"Zone {i+1}", tuple(centroid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Blend with original
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        return frame


def draw_trajectory(frame, trajectory, color=(0, 255, 0), thickness=2):
    """Draw object trajectory on frame."""
    if len(trajectory) < 2:
        return frame
    
    points = np.array(trajectory, dtype=np.int32)
    for i in range(1, len(points)):
        cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, thickness)
    
    # Draw start and end points
    cv2.circle(frame, tuple(points[0]), 5, (0, 255, 0), -1)  # Start (green)
    cv2.circle(frame, tuple(points[-1]), 5, (0, 0, 255), -1)  # End (red)
    
    return frame


def infer_comprehensive(source, model_path='yolo11n.pt', conf_threshold=0.25,
                       track=False, zones=None, save_trajectory=False,
                       analytics=False, export_json=False, export_csv=False,
                       save_output=True, display=True, stationary_alert=False):
    """
    Run comprehensive inference with advanced features.
    
    Args:
        source (str): Path to video or image
        model_path (str): Path to YOLO model
        conf_threshold (float): Confidence threshold
        track (bool): Enable object tracking
        zones (list): List of zone polygons
        save_trajectory (bool): Draw trajectories
        analytics (bool): Show analytics dashboard
        export_json (bool): Export results to JSON
        export_csv (bool): Export results to CSV
        save_output (bool): Save output video
        display (bool): Display video
        stationary_alert (bool): Alert for stationary objects
    
    Returns:
        dict: Results dictionary
    """
    print(f"\n{'='*70}")
    print(f"🚀 COMPREHENSIVE YOLO INFERENCE")
    print(f"{'='*70}")
    print(f"📁 Source: {source}")
    print(f"📦 Model: {model_path}")
    print(f"🎯 Confidence: {conf_threshold}")
    print(f"🔍 Tracking: {'Enabled' if track else 'Disabled'}")
    print(f"🗺️  Zones: {len(zones) if zones else 0}")
    
    # Check if source exists
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source not found: {source}")
    
    # Load model
    print(f"\n📦 Loading model...")
    model = YOLO(model_path)
    
    # Check if image or video
    ext = Path(source).suffix.lower()
    is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    if is_image:
        # Image inference
        print(f"🖼️  Processing image...")
        results = model(source, conf=conf_threshold, verbose=False)
        result = results[0]
        
        annotated = result.plot()
        
        if save_output:
            output_path = str(Path(source).stem) + "_comprehensive" + str(Path(source).suffix)
            cv2.imwrite(output_path, annotated)
            print(f"💾 Saved to: {output_path}")
        
        if display:
            cv2.imshow('Comprehensive Detection', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return {'detections': len(result.boxes)}
    
    # Video inference
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Error opening video: {source}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📊 Video Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    
    # Setup output
    output_path = None
    out = None
    if save_output:
        output_path = str(Path(source).stem) + "_comprehensive.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize trackers
    tracker = ObjectTracker() if track else None
    zone_detector = ZoneDetector(zones) if zones else None
    
    # Statistics
    stats = {
        'total_detections': 0,
        'frame_detections': [],
        'class_counts': defaultdict(int),
        'stationary_alerts': []
    }
    
    print(f"\n🔍 Processing video...")
    print(f"{'='*70}")
    
    frame_count = 0
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference with tracking
        if track:
            results = model.track(frame, conf=conf_threshold, verbose=False, persist=True)
        else:
            results = model(frame, conf=conf_threshold, verbose=False)
        
        result = results[0]
        annotated_frame = result.plot()
        
        # Process detections
        num_detections = len(result.boxes)
        stats['total_detections'] += num_detections
        stats['frame_detections'].append(num_detections)
        
        # Zone detection
        if zone_detector:
            annotated_frame = zone_detector.draw_zones(annotated_frame)
        
        # Process each detection
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            stats['class_counts'][class_name] += 1
            
            bbox = box.xyxy[0].tolist()
            
            # Tracking
            if track and hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id[0])
                tracker.update(track_id, bbox, class_name, frame_count)
                
                # Draw trajectory
                if save_trajectory:
                    trajectory = tracker.get_trajectory(track_id)
                    draw_trajectory(annotated_frame, trajectory)
                
                # Stationary alert
                if stationary_alert and tracker.is_stationary(track_id):
                    cv2.putText(annotated_frame, "STATIONARY!", 
                               (int(bbox[0]), int(bbox[1])-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    stats['stationary_alerts'].append({
                        'frame': frame_count,
                        'track_id': track_id,
                        'class': class_name
                    })
            
            # Zone detection
            if zone_detector:
                zones_in = zone_detector.check_zones(bbox, frame_count)
                if zones_in:
                    zone_text = f"Zones: {','.join(map(str, [z+1 for z in zones_in]))}"
                    cv2.putText(annotated_frame, zone_text,
                               (int(bbox[0]), int(bbox[3])+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Analytics overlay
        if analytics:
            y_offset = 30
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            cv2.putText(annotated_frame, f"Objects: {num_detections}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            cv2.putText(annotated_frame, f"Total: {stats['total_detections']}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if track and tracker:
                y_offset += 25
                cv2.putText(annotated_frame, f"Tracks: {len(tracker.tracks)}",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"⏳ {progress:.1f}% - Frame {frame_count}/{total_frames} - {num_detections} objects")
        
        # Save frame
        if save_output and out:
            out.write(annotated_frame)
        
        # Display
        if display:
            cv2.imshow('Comprehensive Inference', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹️  Stopped by user")
                break
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print(f"\n{'='*70}")
    print(f"✅ PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"📊 Statistics:")
    print(f"   Frames: {frame_count}")
    print(f"   Total detections: {stats['total_detections']}")
    print(f"   Avg detections/frame: {stats['total_detections']/frame_count:.2f}")
    
    print(f"\n📦 Class Distribution:")
    for class_name, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {class_name}: {count}")
    
    if track and tracker:
        print(f"\n🔍 Tracking:")
        print(f"   Total tracks: {len(tracker.tracks)}")
        if stationary_alert:
            print(f"   Stationary alerts: {len(stats['stationary_alerts'])}")
    
    if zone_detector:
        print(f"\n🗺️  Zone Statistics:")
        for zone_id, count in zone_detector.zone_counts.items():
            print(f"   Zone {zone_id+1}: {count} detections")
    
    # Export results
    if export_json:
        json_path = str(Path(source).stem) + "_results.json"
        export_data = {
            'source': source,
            'frames': frame_count,
            'statistics': {
                'total_detections': stats['total_detections'],
                'class_counts': dict(stats['class_counts']),
                'stationary_alerts': stats['stationary_alerts']
            }
        }
        if track and tracker:
            export_data['tracks'] = {
                str(k): {
                    'class': v['class_name'],
                    'first_seen': v['first_seen'],
                    'last_seen': v['last_seen'],
                    'trajectory_length': len(v['positions'])
                }
                for k, v in tracker.get_all_tracks().items()
            }
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\n💾 JSON exported to: {json_path}")
    
    if export_csv:
        csv_path = str(Path(source).stem) + "_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Detections'])
            for i, count in enumerate(stats['frame_detections'], 1):
                writer.writerow([i, count])
        print(f"💾 CSV exported to: {csv_path}")
    
    if save_output and output_path:
        print(f"\n💾 Video saved to: {output_path}")
    
    return stats


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description='Comprehensive YOLO Inference with Advanced Features',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image or video file')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--track', action='store_true',
                       help='Enable object tracking')
    parser.add_argument('--zones', type=str, default=None,
                       help='Zone polygons as JSON string, e.g., "[[100,100,500,100,500,400,100,400]]"')
    parser.add_argument('--save-trajectory', action='store_true',
                       help='Draw object trajectories')
    parser.add_argument('--analytics', action='store_true',
                       help='Show analytics overlay')
    parser.add_argument('--export-json', action='store_true',
                       help='Export results to JSON')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results to CSV')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display output')
    parser.add_argument('--stationary-alert', action='store_true',
                       help='Alert for stationary objects')
    
    args = parser.parse_args()
    
    # Parse zones
    zones = None
    if args.zones:
        try:
            zones = json.loads(args.zones)
        except:
            print(f"❌ Error parsing zones JSON")
            return
    
    # Print header
    print(f"\n{'='*70}")
    print(f"🚀 COMPREHENSIVE YOLO INFERENCE")
    print(f"{'='*70}")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device.upper()}")
    
    # Run inference
    try:
        infer_comprehensive(
            source=args.source,
            model_path=args.model,
            conf_threshold=args.conf,
            track=args.track,
            zones=zones,
            save_trajectory=args.save_trajectory,
            analytics=args.analytics,
            export_json=args.export_json,
            export_csv=args.export_csv,
            save_output=not args.no_save,
            display=not args.no_display,
            stationary_alert=args.stationary_alert
        )
        
        print(f"\n✅ Inference completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
