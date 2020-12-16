# sargen 

![codecov](https://codecov.io/gh/baterflyrity/sargen/branch/master/graph/badge.svg?token=S8US09OMDB)
![Upload code coverage to Codecov](https://github.com/baterflyrity/sargen/workflows/Upload%20code%20coverage%20to%20Codecov/badge.svg)

SAR images generator from optic ones written in python 3.

---

# Generation algorithm

<!--
https://mermaid-js.github.io/mermaid-live-editor

graph LR
    1[/Load optical<br>image/] ==> 2
    subgraph emulate imagery fa:fa-camera  
    2[Convert to<br>grayscale] ==> 3[Add noise]
    end
    subgraph emulate units fa:fa-male
    3 ==> 4[Scale unit] ==> 5[Rotate unit] ==> 6[Place unit]
    end
    subgraph emulate fly fa:fa-plane
    6 ==> Rotate ==> Downscale ==> Tilt 
    end     

-->
[![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICAxWy9Mb2FkIG9wdGljYWw8YnI-aW1hZ2UvXSA9PT4gMlxuICAgIHN1YmdyYXBoIGVtdWxhdGUgaW1hZ2VyeSBmYTpmYS1jYW1lcmEgIFxuICAgIDJbQ29udmVydCB0bzxicj5ncmF5c2NhbGVdID09PiAzW0FkZCBub2lzZV1cbiAgICBlbmRcbiAgICBzdWJncmFwaCBlbXVsYXRlIHVuaXRzIGZhOmZhLW1hbGVcbiAgICAzID09PiA0W1NjYWxlIHVuaXRdID09PiA1W1JvdGF0ZSB1bml0XSA9PT4gNltQbGFjZSB1bml0XVxuICAgIGVuZFxuICAgIHN1YmdyYXBoIGVtdWxhdGUgZmx5IGZhOmZhLXBsYW5lXG4gICAgNiA9PT4gUm90YXRlID09PiBEb3duc2NhbGUgPT0-IFRpbHQgXG4gICAgZW5kICAgICBcbiIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0In0sInVwZGF0ZUVkaXRvciI6ZmFsc2V9)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICAxWy9Mb2FkIG9wdGljYWw8YnI-aW1hZ2UvXSA9PT4gMlxuICAgIHN1YmdyYXBoIGVtdWxhdGUgaW1hZ2VyeSBmYTpmYS1jYW1lcmEgIFxuICAgIDJbQ29udmVydCB0bzxicj5ncmF5c2NhbGVdID09PiAzW0FkZCBub2lzZV1cbiAgICBlbmRcbiAgICBzdWJncmFwaCBlbXVsYXRlIHVuaXRzIGZhOmZhLW1hbGVcbiAgICAzID09PiA0W1NjYWxlIHVuaXRdID09PiA1W1JvdGF0ZSB1bml0XSA9PT4gNltQbGFjZSB1bml0XVxuICAgIGVuZFxuICAgIHN1YmdyYXBoIGVtdWxhdGUgZmx5IGZhOmZhLXBsYW5lXG4gICAgNiA9PT4gUm90YXRlID09PiBEb3duc2NhbGUgPT0-IFRpbHQgXG4gICAgZW5kICAgICBcbiIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0In0sInVwZGF0ZUVkaXRvciI6ZmFsc2V9)

1) ### Load optical image
  
    Load an optical image in high scale (resolution) in RGB mode.

2) ### Convert to grayscale

    Convert colors from RGB space to grayscale space.   
    Supported algorithms:
   
    - Gamma-correction by ITU-R BT.601  
    ![](https://latex.codecogs.com/png.latex?Y=0,299R+0,557G+0,144B)
      
    - HSV space convertation    
    ![](https://latex.codecogs.com/png.latex?Y=max(R;G;B))
      
3) ### Add noise

    Add gaussian noise to each pixel with luma clamping.

4) ### Scale unit
    Adjust unit size to image scale.

5) ### Rotate unit
    Emulate different orientation by random rotation.

6) ### Place unit
    Emulate different site by random position.  
    **Note that now unit can be places anywhere on the whole image.**

7) ### Rotate image
    Emulate different fly horizontal direction by random rotation.

8) ### Downscale
    Emulate different fly height by random downscaling.

9) ### Tilt
    Emulate different fly vertical direction by random tilt (assuming radar has hard joint).