# Video-Analysis


The Screen recording from Loop11 or OSG Search is processed through this code
- Converts Video into Images based on FPS (Frames per Second)
- Looks for Consecutive Frames which are not unique (indicating change in screen - match rate is 80%)
- For each frame, OCR extracts the Text
- From each Text file, we look for No. of Occurances of Brand name
- Then convert to 5 second bucket


**IMP:** Need to integrate the translation text from Google API

