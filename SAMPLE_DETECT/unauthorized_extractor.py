"""
Unauthorized Activity Extractor
Extracts and saves frames/images with unauthorized activities
"""

import cv2
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import json
from config import UNAUTHORIZED_DIR, RESULTS_DIR


class UnauthorizedExtractor:
    """Extract and save unauthorized activity detections"""
    
    def __init__(self):
        self.output_dir = UNAUTHORIZED_DIR
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_from_image(self, image_path: str, unauthorized_detections: List[Dict], 
                          alerts: List[Dict]) -> Dict:
        """
        Extract unauthorized activities from an image
        
        Args:
            image_path: Path to the source image
            unauthorized_detections: List of unauthorized detection dictionaries
            alerts: List of alert dictionaries
        
        Returns:
            Dictionary with extraction results
        """
        if not unauthorized_detections:
            return {
                'success': False,
                'message': 'No unauthorized activities detected',
                'extracted_count': 0
            }
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'message': 'Failed to read image',
                'extracted_count': 0
            }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extracted_files = []
        
        # Save full image with annotations
        full_image_path = self.output_dir / f"unauthorized_full_{timestamp}.jpg"
        cv2.imwrite(str(full_image_path), image)
        extracted_files.append(str(full_image_path))
        
        # Extract individual unauthorized activity regions
        for idx, detection in enumerate(unauthorized_detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Crop region
            cropped = image[y1:y2, x1:x2]
            
            # Save cropped image
            crop_path = self.output_dir / f"unauthorized_crop_{timestamp}_{idx}.jpg"
            cv2.imwrite(str(crop_path), cropped)
            extracted_files.append(str(crop_path))
        
        # Save alert report
        report_path = self.output_dir / f"alert_report_{timestamp}.json"
        report_data = {
            'timestamp': timestamp,
            'source_image': image_path,
            'unauthorized_count': len(unauthorized_detections),
            'alerts': alerts,
            'detections': [
                {
                    'class': d['class'],
                    'confidence': float(d['confidence']),
                    'bbox': [float(x) for x in d['bbox']]
                }
                for d in unauthorized_detections
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        extracted_files.append(str(report_path))
        
        return {
            'success': True,
            'message': f'Extracted {len(unauthorized_detections)} unauthorized activities',
            'extracted_count': len(unauthorized_detections),
            'files': extracted_files,
            'report_path': str(report_path)
        }
    
    def extract_from_video(self, video_path: str, frame_detections: List[Dict]) -> Dict:
        """
        Extract frames with unauthorized activities from video
        
        Args:
            video_path: Path to the source video
            frame_detections: List of dictionaries with frame_number, unauthorized_detections, and alerts
        
        Returns:
            Dictionary with extraction results
        """
        # Filter frames with unauthorized activities
        unauthorized_frames = [f for f in frame_detections if f['unauthorized_detections']]
        
        if not unauthorized_frames:
            return {
                'success': False,
                'message': 'No unauthorized activities detected in video',
                'extracted_count': 0
            }
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'success': False,
                'message': 'Failed to open video',
                'extracted_count': 0
            }
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extracted_files = []
        
        # Extract frames
        for frame_data in unauthorized_frames:
            frame_number = frame_data['frame_number']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Calculate timestamp in video
                time_seconds = frame_number / fps
                time_str = f"{int(time_seconds//60):02d}:{int(time_seconds%60):02d}"
                
                # Save frame
                frame_path = self.output_dir / f"unauthorized_frame_{timestamp}_f{frame_number}_{time_str.replace(':', '-')}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted_files.append(str(frame_path))
        
        cap.release()
        
        # Save comprehensive report
        report_path = self.output_dir / f"video_alert_report_{timestamp}.json"
        report_data = {
            'timestamp': timestamp,
            'source_video': video_path,
            'total_frames_analyzed': len(frame_detections),
            'frames_with_unauthorized': len(unauthorized_frames),
            'fps': fps,
            'frames': [
                {
                    'frame_number': f['frame_number'],
                    'time_seconds': f['frame_number'] / fps,
                    'unauthorized_count': len(f['unauthorized_detections']),
                    'alerts': f['alerts']
                }
                for f in unauthorized_frames
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        extracted_files.append(str(report_path))
        
        return {
            'success': True,
            'message': f'Extracted {len(unauthorized_frames)} frames with unauthorized activities',
            'extracted_count': len(unauthorized_frames),
            'files': extracted_files,
            'report_path': str(report_path)
        }
    
    def get_recent_extractions(self, limit: int = 10) -> List[str]:
        """Get list of recent unauthorized activity extractions"""
        files = list(self.output_dir.glob("unauthorized_*.jpg"))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return [str(f) for f in files[:limit]]
