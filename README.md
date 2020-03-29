<h3> Diameter Estimation for carotid artery (CCA) ultrasound images </h3>

Prerequisites: <br/>
<ul>
  <li> Set data file paths in path.py</li>
  <li> Results are observed in corresponding data folder [/data/us_images_resized/output.txt]
</ul>

<h4>
How to run:
</h4>
  train.py --backbone="resnet18" <br/>
  [refer train.py for other params]
  

<p>
To try: <br />
1) Set LR scheduler
2) resize after crop
3) try another model
4) k-fold cross validation
</p>

For now only success when:
RESIZE = True and CROPPED = False
lr = 0.001
