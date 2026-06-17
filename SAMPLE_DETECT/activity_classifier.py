"""
Activity Classifier for Parks Surveillance
Classifies detections as authorized or unauthorized activities
"""

import numpy as np
from typing import List, Dict, Tuple
from config import (
    UNAUTHORIZED_ACTIVITIES,
    AUTHORIZED_ACTIVITIES,
    SEVERITY_LEVELS,
    PROXIMITY_THRESHOLD
)


class ActivityClassifier:
    """Classifies detected objects as authorized or unauthorized activities"""
    
    def __init__(self):
        self.unauthorized_categories = UNAUTHORIZED_ACTIVITIES
        self.authorized_classes = AUTHORIZED_ACTIVITIES
        self.severity_levels = SEVERITY_LEVELS
        self.proximity_threshold = PROXIMITY_THRESHOLD
    
    def classify_detections(self, detections: List[Dict]) -> Dict:
        """
        Classify all detections into authorized and unauthorized activities
        
        Args:
            detections: List of detection dictionaries with keys:
                - class: detected class name
                - confidence: detection confidence
                - bbox: bounding box coordinates [x1, y1, x2, y2]
        
        Returns:
            Dictionary with classification results
        """
        authorized = []
        unauthorized = []
        alerts = []
        
        # Extract person and bag positions for proximity analysis
        persons = [d for d in detections if d['class'] == 'person']
        bags = [d for d in detections if d['class'] == 'Bag']
        
        for detection in detections:
            class_name = detection['class']
            
            # Check for vandalism
            if class_name in self.unauthorized_categories['vandalism']:
                severity = self._get_severity(class_name)
                unauthorized.append(detection)
                alerts.append({
                    'type': 'Vandalism Detected',
                    'class': class_name,
                    'severity': severity,
                    'confidence': detection['confidence'],
                    'description': f"Potential vandalism activity detected: {class_name}"
                })
            
            # Check for unattended items
            elif class_name in self.unauthorized_categories['suspicious_items']:
                if not self._is_item_attended(detection, persons):
                    severity = self._get_severity(class_name)
                    unauthorized.append(detection)
                    alerts.append({
                        'type': 'Unattended Item',
                        'class': class_name,
                        'severity': severity,
                        'confidence': detection['confidence'],
                        'description': f"Unattended {class_name} detected - no person nearby"
                    })
                else:
                    authorized.append(detection)
            
            # Check for vehicle violations
            elif class_name in self.unauthorized_categories['vehicle_violations']:
                severity = self._get_severity(class_name)
                unauthorized.append(detection)
                alerts.append({
                    'type': 'Vehicle Violation',
                    'class': class_name,
                    'severity': severity,
                    'confidence': detection['confidence'],
                    'description': f"{class_name} detected in restricted area"
                })
            
            # Authorized activities
            else:
                authorized.append(detection)
        
        return {
            'authorized': authorized,
            'unauthorized': unauthorized,
            'alerts': alerts,
            'summary': {
                'total_detections': len(detections),
                'authorized_count': len(authorized),
                'unauthorized_count': len(unauthorized),
                'alert_count': len(alerts),
                'high_severity': len([a for a in alerts if a['severity'] == 'HIGH']),
                'medium_severity': len([a for a in alerts if a['severity'] == 'MEDIUM']),
                'low_severity': len([a for a in alerts if a['severity'] == 'LOW'])
            }
        }
    
    def _is_item_attended(self, item_detection: Dict, persons: List[Dict]) -> bool:
        """
        Check if an item (like a bag) is attended by checking proximity to persons
        
        Args:
            item_detection: Detection dictionary for the item
            persons: List of person detections
        
        Returns:
            True if item is near a person, False otherwise
        """
        if not persons:
            return False
        
        item_bbox = item_detection['bbox']
        item_center = self._get_bbox_center(item_bbox)
        
        for person in persons:
            person_bbox = person['bbox']
            person_center = self._get_bbox_center(person_bbox)
            
            # Calculate distance between centers
            distance = np.sqrt(
                (item_center[0] - person_center[0])**2 +
                (item_center[1] - person_center[1])**2
            )
            
            if distance < self.proximity_threshold:
                return True
        
        return False
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _get_severity(self, class_name: str) -> str:
        """Get severity level for a class"""
        for severity, classes in self.severity_levels.items():
            if class_name in classes:
                return severity
        return 'LOW'
    
    def get_activity_description(self, class_name: str) -> str:
        """Get human-readable description of activity"""
        descriptions = {
            'Drawing': 'Graffiti/Drawing on property',
            'Chalkpiece': 'Chalk marking/vandalism',
            'Bag': 'Unattended bag or package',
            'car': 'Vehicle in restricted area',
            'Bicycle': 'Bicycle in restricted zone',
            'person': 'Person in area',
            'Bench': 'Park bench',
            'Tree': 'Tree',
            'Building': 'Building structure',
            'Cap': 'Personal item (cap)',
            'road': 'Road/pathway'
        }
        return descriptions.get(class_name, class_name)
