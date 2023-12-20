lint: FORCE
	./scripts/lint.sh

format:
	./scripts/clean.sh

test: FORCE 
	./scripts/test.sh
	./scripts/test_notebooks.sh
	

FORCE: