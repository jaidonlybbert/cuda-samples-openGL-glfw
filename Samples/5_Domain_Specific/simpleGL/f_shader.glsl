#version 440 core

out vec4 FragColor;
  
in vec4 vertexColor;

void main()
{
    FragColor = vertexColor;//vec4(1.0, 0, 0, 1);
} 