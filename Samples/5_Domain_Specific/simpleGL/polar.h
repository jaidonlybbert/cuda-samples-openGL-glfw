#pragma once

#define CLAMP_TO_ZERO 0.001

glm::vec3 cartesianToPolarPoint(glm::vec3 cartesian) {
	// Expected cartesian coordinates of form (x, y, z)

	// Convention: 
	// r (0 to inf) = radius
	// theta (0 to 180) = (+z to -z)
	// phi (0 to 360) = (+x to +x)
	float r = glm::sqrt(cartesian.x * cartesian.x + cartesian.y * cartesian.y + cartesian.z * cartesian.z);

	float theta;
	float phi;

	if (abs(cartesian.z) <= CLAMP_TO_ZERO) { // Avoid divide by 0
		theta = glm::pi<GLfloat>() / 2;
	}
	else if (abs(cartesian.x) <= CLAMP_TO_ZERO && abs(cartesian.y) <= CLAMP_TO_ZERO) {
		if (cartesian.z > 0) {
			theta = 0;
		}
		else if (cartesian.z < 0) {
			theta = glm::pi<GLfloat>();
		}
		else {
			theta = 0;
		}
	}
	else if (cartesian.z > 0) {
		theta = glm::atan(glm::sqrt(cartesian.x * cartesian.x + cartesian.y * cartesian.y) / cartesian.z);
	}
	else if (cartesian.z < 0) {
		theta = (glm::pi<GLfloat>() / 2) + glm::atan(abs(cartesian.z) / glm::sqrt(cartesian.x * cartesian.x + cartesian.y * cartesian.y));
	}
	else {
		theta = 0;
	}

	if (cartesian.x == 0) {
		if (cartesian.y > 0) {
			phi = glm::pi<GLfloat>() / 2;
		}
		else if (cartesian.y < 0) {
			phi = glm::pi<GLfloat>() * 3 / 2;
		}
		else {
			phi = 0;
		}
	}
	else if (cartesian.y == 0) {
		if (cartesian.x > 0) {
			phi = 0;
		}
		else if (cartesian.x < 0) {
			phi = glm::pi<GLfloat>();
		}
		else {
			phi = 0;
		}
	}
	else if (cartesian.y > 0 && cartesian.x > 0) {
		phi = glm::atan(cartesian.y / cartesian.x);
	}
	else if (cartesian.y > 0 && cartesian.x < 0) {
		phi = glm::pi<GLfloat>() - glm::atan(cartesian.y / abs(cartesian.x));
	}
	else if (cartesian.y < 0 && cartesian.x < 0) {
		phi = glm::pi<GLfloat>() + glm::atan(cartesian.y / cartesian.x);
	}
	else if (cartesian.y < 0 && cartesian.x > 0) {
		phi = 2 * glm::pi<GLfloat>() - glm::atan(abs(cartesian.y) / cartesian.x);
	}
	else {
		phi = 0;
	}

	return glm::vec3(r, theta, phi);

	//vector transformation matrix
	//glm::mat3 cartesianToPolar(glm::vec3(glm::sin(theta) * glm::cos(phi), glm::sin(theta) * glm::sin(phi), glm::cos(theta)),
	//	glm::vec3(glm::cos(theta) * glm::cos(phi), glm::cos(theta) * glm::sin(phi), -glm::sin(theta)),
	//	glm::vec3(-glm::sin(phi), glm::cos(phi), 0.0f));
}

glm::vec3 polarToCartesianPoint(glm::vec3 polar) {
	// Expected polar coordinates of form (r, theta, phi)

	// Convention: 
	// r (0 to inf) = radius
	// theta (0 to 180) = (+z to -z)
	// phi (0 to 360) = (+x to +x)

	float r = polar[0];
	float theta = polar[1];
	float phi = polar[2];

	glm::vec3 cartesian(0);
	cartesian.x = r * glm::sin(theta) * glm::cos(phi);
	cartesian.y = r * glm::sin(theta) * glm::sin(phi);
	cartesian.z = r * glm::cos(theta);

	return cartesian;
}