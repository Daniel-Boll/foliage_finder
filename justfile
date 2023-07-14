# vim: set ft=make ts=4 sw=4 noet:

alias r := run

# Run the project
run: clean
	poetry run python src/main.py

# Run tests
test:
	poetry run pytest tests/

simple:
	poetry run python teste.py

clean:
	rm -rf outputs/contours/*
	rm -rf classified/*

show:
	brave classified/*
