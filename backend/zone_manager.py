import cv2
import numpy as np

def point_inside_zone(point, zone, width, height):
    polygon = np.array([[int(x*width), int(y*height)] for x,y in zone["points"]], dtype=np.int32)
    if len(polygon) < 3:
        return False
    return cv2.pointPolygonTest(polygon.reshape((-1,1,2)), (float(point[0]),float(point[1])), False) >= 0

def draw_zones(frame, zones):
    h,w = frame.shape[:2]
    colors = {"low":(0,200,0), "medium":(0,200,255), "high":(0,120,255), "critical":(0,0,255)}
    for zone in zones:
        pts = np.array([[int(x*w),int(y*h)] for x,y in zone["points"]], dtype=np.int32)
        if len(pts) < 3: continue
        color = colors.get(zone.get("severity","high"), (0,0,255))
        overlay = frame.copy()
        cv2.fillPoly(overlay,[pts],color)
        cv2.addWeighted(overlay,0.18,frame,0.82,0,frame)
        cv2.polylines(frame,[pts],True,color,2)
        x,y = pts[0]
        cv2.putText(frame,zone["name"],(x,max(22,y-8)),cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2)
    return frame
