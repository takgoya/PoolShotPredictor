# PoolShotPredictor
CVZone Pool Shot Predictor contest

## DESCRIPTION: 
The steps taken are described: 

1) Find the table using HSV space
2) Using contours, the pocket holes are found
3) Cue ball and stick is found by contours too
4) The target ball is found using HoughCircles
5) A line is predicted from the stick along the cue ball to simulate the path of the cue ball, and is calculated a possible collision with the target ball
6) If the collision is found, it calculates the target path as the line created from the cue ball towards the target ball (straight line passing the centers)
7) On that target ball path is predicted the final position, if it reaches a pocket hole or not. In case of not reaching a pocket hole, its reflection is calculated to check if there is a new predicted final point
8) The final prediction is drawn on the table, in green if it is IN or red if it is OUT
