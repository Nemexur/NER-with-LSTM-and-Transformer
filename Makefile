.PHONY: experiments
experiments:
	bash bin/experiments.sh

.PHONY: clean
clean:
	docker rmi ner_experiments:dev --force
