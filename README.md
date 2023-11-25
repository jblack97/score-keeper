# SCORE-KEEPER
This project aims to provide an end-to-end pipeline for validating the results of bets shown in an image of a betslip. This problem contains elements of OCR (extracting the text from screenshots of betslips), NLP (parsing the OCR output into distinct bets), and webscraping (validating the results of the bets).

This is still a work in progress. The OCR solution works well for screenshots of betslips, and GPT-3.5 can solve the NLP task with only a handful examples. I am currently working on generating a synthetic dataset to train my own tagger model to remove the OpenAI dependency, the code for this can be found in **src/bet_identification/betslip_tagger**.

# Input 
Any image containing the information from a betslip e.g. \
<img src="https://github.com/jblack97/score-keeper/blob/main/betslip.jpg" width="500" height="750" >

# Betslip Scanning (Computer Vision)
We then use opencv's contour recognition tool to recognize text in the image. We need to extract the text line-by-line in order to get the most accurate results from our text recognition tool. To identify the lines of text, I used DBScan Clustering. The results of this are displayed below, where each colour represents a cluster of contours. By taking the min and max height values of each cluster, we can extract individual jpeg's for each line, and feed them into our text recognition tool.
 <img src="https://github.com/jblack97/score-keeper/blob/main/clustered_contours_betslip.jpg" width="500" height="750" > \
Pytesseract then does a great job of reading the words into our pipeline.

# Bet Identification (NLP)
The next task is to parse the text into a set of bets than can be individually verified. Each bet has an associated Event, Market, Side, and Odds. inference.py currently uses the OpenAI api to call GPT3.5 to do this, but work is in progress to fine-tune a BERT-variant to do this task. I am currently writing the code for the generation of a synthetic training set. The code for this can be found in **src/bet_identification/betslip_tagger**.

# Output
```
[
            {
                "event": "Whittaker, Robert - Du Plessis, Dricus",
                "market": "Bout Odds",
                "side": "Whittaker, Robert",
                "odds": "1/4",
                "extra": []
            },
            {
                "event": "Volkanovski, Alex - Rodriguez, Yair",
                "market": "Bout Odds",
                "side": "Volkanovski, Alex",
                "odds": "6/25",
                "extra": []
            },
            {
                "event": "Moreno, Brandon - Pantoja, Alexandre",
                "market": "Bout Odds",
                "side": "Moreno, Brandon",
                "odds": "12/25",
                "extra": []
            },
            {
                "event": "Lawler, Robbie - Price, Niko",
                "market": "Bout Odds",
                "side": "Price, Niko",
                "odds": "21/50",
                "extra": []
            }
        ]
```


