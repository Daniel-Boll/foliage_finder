# vim: set ft=make ts=4 sw=4 noet:

alias r := run

# Run the project
run:
	poetry run python src/main.py

# Run tests
test:
	poetry run pytest tests/

simple:
	poetry run python teste.py

clean:
	rm -rf contour_image.png
	find outputs/contours -type f -name '*.png' -delete

show:
	feh contour_image.png
