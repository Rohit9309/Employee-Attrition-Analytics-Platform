# IBM Watson Studio / Watson Machine Learning deployment guide

This guide provides a deployment path, not a claim that the model is already deployed. IBM Cloud service names, available runtimes, and console steps can change; confirm the current IBM documentation and your account's available services.

## A. Prepare the model
1. Run `python train_models.py` locally and inspect `outputs/model_metrics.csv`.
2. Confirm `models/best_model.joblib` and `models/model_metadata.json` exist.
3. Keep a record of the dataset version, feature schema, train/test split, package versions, and evaluation metrics.
4. For production, package preprocessing and estimator together (the project does this with a scikit-learn Pipeline).

## B. Use Watson Studio
1. Sign in to IBM Cloud and open Watson Studio.
2. Create or select a project.
3. Add the dataset as a project asset, subject to your institution's data/privacy requirements.
4. Import `notebooks/advanced_model_comparison.ipynb` or the training script into the project.
5. Run the notebook in an environment that supports the required Python packages. If a package is unavailable in the managed runtime, adapt the environment and document the version changes.

## C. Deploy with Watson Machine Learning
1. Create/select a Watson Machine Learning service instance supported by your IBM Cloud account.
2. Follow the current IBM documentation for registering the trained model or training asset.
3. Create a deployment using a supported runtime and model format.
4. Send a test scoring request using the exact feature schema and verify the response against local predictions.
5. Record the deployment ID, endpoint URL, runtime, model version, and test results in `docs/api_documentation.md`.

Do not commit API keys, IAM tokens, service credentials, or private endpoint details to GitHub. Store secrets in environment variables or an approved secret manager.

## D. REST endpoint
The local FastAPI reference endpoint is in `api/main.py`. Run:
```bash
uvicorn api.main:app --reload
```
Visit `http://127.0.0.1:8000/docs`. This local endpoint is not an IBM-hosted endpoint. To use Watson Machine Learning, follow the current IBM scoring endpoint request schema and adapt the client to the deployed asset.

## E. Scheduled retraining with Watson Studio Jobs
1. Put data validation, training, evaluation, and artifact export into a repeatable notebook or script.
2. Create a Watson Studio job using the currently supported job type and runtime in your project.
3. Set a schedule that matches the data refresh cadence.
4. Add a quality gate: do not deploy automatically if required columns are missing, target labels are invalid, test performance degrades, or fairness checks fail.
5. Save run logs and metrics; require approval before promoting a new model to deployment.
6. Test rollback to the prior known-good model.

## F. Deployment checklist
- [ ] Schema validation and missing-value handling
- [ ] Held-out test evaluation
- [ ] Group-wise error/fairness review where lawful and appropriate
- [ ] Human review and documented limitations
- [ ] Secrets stored outside Git
- [ ] Endpoint authentication and rate limiting
- [ ] Monitoring, versioning, rollback and retraining logs
