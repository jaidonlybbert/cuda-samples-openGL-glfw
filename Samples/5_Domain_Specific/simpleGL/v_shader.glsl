#version 440 core

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec4 vColor;

// MVP transforms
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

out vec4 vertexColor;

void main() 
{
	gl_Position = proj * view * model * vec4(vPosition, 1.0);
	vertexColor = vColor;
}
