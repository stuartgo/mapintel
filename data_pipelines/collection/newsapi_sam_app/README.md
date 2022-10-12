# NewsAPI SAM App
AWS SAM App to retrieve news data from NewsAPI on a schedule (8 in 8 hours) and save them to a S3 bucket.
By deploying this app, you will create the following resources:
- Lambda Function - runs the Python function to retrieve the news
- S3 Bucket - stores the news data as json files
- SNS Topic - used to notify the administrator if the function fails

## Usage:
### Prerequisites:
1. Creating an AWS account.
2. Configuring AWS Identity and Access Management (IAM) permissions.
3. Installing the AWS SAM command line interface (CLI). Note: Make sure that you have version 1.13.0 or later. Check the version by running the sam --version command.

### Step 1 - Build the application:
```
cd newsapi_sam_app
sam build
```

### Step 2 - Deploy the application:
```
sam deploy --guided
```
, and follow the on-screen prompts.

## References:
- [https://docs.aws.amazon.com/lambda/latest/dg/lambda-settingup.html#lambda-settingup-awssam](https://docs.aws.amazon.com/lambda/latest/dg/lambda-settingup.html#lambda-settingup-awssam)
- [https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-getting-started-hello-world.html](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-getting-started-hello-world.html)
