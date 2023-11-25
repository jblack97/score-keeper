# SCORE-KEEPER
This project aims to provide an end-to-end pipeline for validating the results of bets shown in an image of a betslip. This problem contains elements of OCR (extracting the text from screenshots of betslips), NLP (parsing the OCR output into distinct bets), and webscraping (validating the results of the bets).

This is still a work in progress. The OCR solution works well for screenshots of betslips, and GPT-3.5 can solve the NLP task with only a handful examples. I am currently working on generating a synthetic dataset to train my own tagger model to remove the OpenAI dependency, the code for this can be found in **src/bet_identification/betslip_tagger**.

# Input 
Any image containing the information from a betslip e.g. ![alt text](https://github.com/jblack97/score-keeper/blob/main/betslip.jpg)

# Text Recognition
We then use opencv's contour recognition tool to recognize text in the image. We need to extract the text line-by-line in order to get the most accurate results from our text recognition tool. To identify the lines of text, I used DBScan Clustering. The results of this are displayed below, where each colour represents a cluster of contours. By taking the min and max height values of each cluster, we can extract individual jpeg's for each line, and feed them into our text recognition tool.
![alt text](https://github.com/jblack97/score-keeper/blob/main/clustered_contours_betslip.jpg)
