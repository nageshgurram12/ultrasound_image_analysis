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
  

<h3>
RESUTLS:
</h3>

<hr/>
<table>
<tr>
<th>Augmentation by Crop</th>
<th>Resized</th>
<th>Cross Validation</th>
<th>Model</th>
<th>Train Loss</th>
<th>Validation Loss</th>
<th>Test Loss</th>
</tr>

<tr>
  <td><span>&#10003;</span></td>
  <td><span>&#10060;</span></td>
  <td><span>&#10060;</span></td>
  <td>Resnet18</td>
</tr>
</table>

<p>
ToDo: <br />
- Try ChextNet pretrained model

 Set LR scheduler - Not much improvement <br/>
 resize after crop - Not much improvement <br/>
 try another model - Done <br/>
 k-fold cross validation - Done <br/>
</p>

<!-- Comments:
For now only success when:
RESIZE = True and CROPPED = False
lr = 0.001
-->
