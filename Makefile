# Public variables to be set by the user in the Makefile
PLUGIN_VERSION=1.0.0
DSS_VERSION=7.0
PLUGIN_ID=google-cloud-nlp

# Private variables to be set by the user in the environment
ifndef DKU_PLUGIN_DEVELOPER_ORG
$(error the DKU_PLUGIN_DEVELOPER_ORG environment variable is not set)
endif
ifndef DKU_PLUGIN_DEVELOPER_TOKEN
$(error the DKU_PLUGIN_DEVELOPER_TOKEN environment variable is not set)
endif
ifndef DKU_PLUGIN_ARTIFACT_REPO_URL
$(error the DKU_PLUGIN_ARTIFACT_REPO_URL environment variable is not set)
endif


# evaluate additional variable
archive_file_name="dss-plugin_${PLUGIN_ID}_${PLUGIN_VERSION}.zip"
remote_url=`git config --get remote.origin.url`
last_commit_id=`git rev-parse HEAD`

plugin:
	@echo "[START] Archiving plugin to dist/ folder..."
	cat plugin.json | json_pp > /dev/null
	rm -rf dist
	mkdir dist
	echo "{\"remote_url\":\"${remote_url}\",\"last_commit_id\":\"${last_commit_id}\"}" > dist/release_info.json
	git archive -9 --format zip -o dist/${archive_file_name} HEAD
	zip -ur dist/${archive_file_name} dist/release_info.json
	@echo "[SUCCESS] Archiving plugin to dist/ folder: Done!"

publish: plugin
	@echo "[START] Publishing archive to artifact repository..."
	@echo "[SUCCESS] Publishing archive to artifact repository: Done!"


dist-clean:
	rm -rf dist

