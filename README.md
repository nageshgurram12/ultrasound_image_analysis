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
<th>Prediction</th>
<th>Resized</th>
<th>Cross Validation</th>
<th>Model</th>
<th>Test Loss</th>
</tr>

<tr>
  <td><span>&#10003;</span></td>
  <td><span>Center Diameter</span></td>
  <td><span>&#10003;</span></td>
  <td><span>&#10060;</span></td>
  <td>Resnet18</td>
  <td>0.13</td>
  <td>
</tr>

<tr>
  <td><span>&#10003;</span></td>
  <td><span>Center Diameter</span></td>
  <td><span>&#10060;</span></td>
  <td><span>&#10060;</span></td>
  <td>Resnet18</td>
  <td>0.34</td>
  <td>
</tr>

<tr>
  <td><span>&#10003;</span></td>
  <td><span>Average</span></td>
  <td><span>&#10003;</span></td>
  <td><span>&#10060;</span></td>
  <td>Resnet18</td>
  <td>0.25</td>
  <td>
</tr>

<tr>
  <td><span>&#10003;</span></td>
  <td><span>Average</span></td>
  <td><span>&#10060;</span></td>
  <td><span>&#10060;</span></td>
  <td>Resnet18</td>
  <td>0.24</td>
  <td>
</tr>

<tr>
  <td><span>&#10003;</span></td>
  <td><span>Center Diameter</span></td>
  <td><span>&#10003;</span></td>
  <td><span>&#10003;</span></td>
  <td>Resnet18</td>
  <td>0.11</td>
</tr>

</table>

<hr />

<p>
To analyse the results and generate plots, run 'analyse_cv_results.py' by passing data from './data/us_images_resized/analyse.txt'.
All the data that is used to generate these plots are from this file.
</p>

<hr />
<p>
The list of images with high error (in Fig.6): 
<table>
<tr>
<th>Image name</th>
<th>Error </th>
</tr>
<tr>
<td>12-06-36.jpg</td>
<td>2.44</td>
</tr>
<tr>
<td>11-12-47.jpg</td>
<td>1.59</td>
</tr>
<tr>
<td>09-53-51_2.jpg</td>
<td>1.34</td>
</tr>
</table>
</p>