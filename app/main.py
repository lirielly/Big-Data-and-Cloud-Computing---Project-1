# Imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import flask
import logging
import os
import tfmodel
import numpy as np
from random import shuffle
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import vision

# Set up logging
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s',
                     datefmt='%Y-%m-%d %H:%M:%S')

PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT') 
logging.info('Google Cloud project is {}'.format(PROJECT))

# Initialisation
logging.info('Initialising app')
app = flask.Flask(__name__)

logging.info('Initialising BigQuery client')
BQ_CLIENT = bigquery.Client()

BUCKET_NAME = PROJECT + '.appspot.com'
logging.info('Initialising access to storage bucket {}'.format(BUCKET_NAME))
APP_BUCKET = storage.Client().bucket(BUCKET_NAME)

logging.info('Initialising TensorFlow classifier')
TF_CLASSIFIER = tfmodel.Model(
    app.root_path + "/static/tflite/model.tflite",
    app.root_path + "/static/tflite/dict.txt"
)
logging.info('Initialisation complete')

# End-point implementation
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/classes')
def classes():
    results = BQ_CLIENT.query(
    '''
        Select Description, COUNT(*) AS NumImages
        FROM `bdcc22project.openimages.image_labels`
        JOIN `bdcc22project.openimages.classes` USING(Label)
        GROUP BY Description
        ORDER BY Description
    ''').result()
    logging.info('classes: results={}'.format(results.total_rows))
    data = dict(results=results)
    return flask.render_template('classes.html', data=data)

@app.route('/relations')
def relations():
    results = BQ_CLIENT.query(
    '''
    SELECT distinct Relation, 
        count(Relation)
    FROM `bdcc22project.openimages.relations` 
    group by Relation
    order by Relation
    ''').result()
    data = dict(results=results)
    return flask.render_template('relations.html', data=data)
  

@app.route('/image_info')
def image_info():
    image_id = flask.request.args.get('image_id')
    results = BQ_CLIENT.query(
    '''
        SELECT distinct 
            FORMAT('%T', ARRAY_AGG(DISTINCT CONCAT(b.Description, '#', d.Relation, '#', c.Description))) AS array_agg,
            FORMAT('%T', ARRAY_AGG(DISTINCT e.Description)) AS array_agg
        FROM `bdcc22project.openimages.image_labels` AS a
        LEFT JOIN `bdcc22project.openimages.relations` AS d ON (a.ImageId=d.ImageId)
        LEFT JOIN `bdcc22project.openimages.classes` AS b ON (d.Label1=b.Label)
        LEFT JOIN `bdcc22project.openimages.classes` AS c ON (d.Label2=c.Label) 
        LEFT JOIN `bdcc22project.openimages.classes` AS e ON (a.Label=e.Label)
        WHERE a.ImageId = '{0}'
    
    '''.format(image_id)
    ).result()

    for row in results:
        results2 = row[1]
        results3 = row[0].strip("[]").replace('"', "").split(", ")
    results4 = list(map(lambda x: x.split("#"), results3))
    
    data = dict(image_id=image_id, 
            results=str(results2),
            results1=results4)
    return flask.render_template('image_info.html', data=data)

@app.route('/image_search')
def image_search():
    description = flask.request.args.get('description')
    image_limit = flask.request.args.get('image_limit', default=10, type=int)
    results = BQ_CLIENT.query(
    '''
        SELECT ImageId
        FROM `bdcc22project.openimages.image_labels`
        JOIN `bdcc22project.openimages.classes` USING(Label)
        WHERE Description = '{0}' 
        ORDER BY ImageId
        LIMIT {1}  
    '''.format(description, image_limit)
    ).result()
    logging.info('image_search: description={} limit={}, results={}'\
           .format(description, image_limit, results.total_rows))
    data = dict(description=description, 
                image_limit=image_limit,
                results=results)
    return flask.render_template('image_search.html', data=data)

@app.route('/relation_search')
def relation_search():
    class1 = flask.request.args.get('class1', default='%')
    relation = flask.request.args.get('relation', default='%')
    class2 = flask.request.args.get('class2', default='%')
    image_limit = flask.request.args.get('image_limit', default=10, type=int)
    results = BQ_CLIENT.query(
    '''
    SELECT  ImageId, 
        b.Description, 
        Relation, 
        c.Description
    FROM `bdcc22project.openimages.relations` AS a
    LEFT JOIN `bdcc22project.openimages.classes` AS b ON (a.Label1=b.Label)
    LEFT JOIN `bdcc22project.openimages.classes` AS c ON (a.Label2=c.Label)
    WHERE Relation LIKE "{1}" and b.Description LIKE '{0}' and c.Description LIKE '{2}'
    group by ImageId,
        b.Description, 
        Relation,
        c.Description
    LIMIT {3} 
    '''.format(class1, relation, class2, image_limit)
    ).result()
    logging.info('classes: results={}'.format(results.total_rows))
    data = dict(class1=class1, 
                relation=relation,
                class2=class2,
                image_limit=image_limit,
                results=results)
    return flask.render_template('relation_search.html', data=data)

@app.route('/image_search_multiple')
def image_search_multiple():
    descriptions = flask.request.args.get('descriptions').split(',')
    image_limit = flask.request.args.get('image_limit', default=10, type=int)
    results = BQ_CLIENT.query(
    '''
    SELECT imageID, array_agg(Description), COUNT(*) AS num1, 
    (SELECT COUNT(*) FROM UNNEST({0})) AS num2
    FROM `bdcc22project.openimages.image_labels` 
    JOIN `bdcc22project.openimages.classes` USING(Label)
    WHERE Description IN  (SELECT * FROM UNNEST({0}))
    GROUP BY imageID
    ORDER BY num1 DESC, imageID
    LIMIT {1}
    
    '''.format(descriptions, image_limit) 
    ).result()

    data = dict(descriptions=descriptions,
                image_limit=image_limit,
                results=results)
    return flask.render_template('image_search_multiple.html', data=data)

@app.route('/image_classify_classes')
def image_classify_classes():
    with open(app.root_path + "/static/tflite/dict.txt", 'r') as f:
        data = dict(results=sorted(list(f)))
        return flask.render_template('image_classify_classes.html', data=data)
 
@app.route('/image_classify', methods=['POST'])
def image_classify():
    files = flask.request.files.getlist('files')
    min_confidence = flask.request.form.get('min_confidence', default=0.25, type=float)
    results = []
    if len(files) > 1 or files[0].filename != '':
        for file in files:
            classifications = TF_CLASSIFIER.classify(file, min_confidence)
            blob = storage.Blob(file.filename, APP_BUCKET)
            blob.upload_from_file(file, blob, content_type=file.mimetype)
            blob.make_public()
            logging.info('image_classify: filename={} blob={} classifications={}'\
                .format(file.filename,blob.name,classifications))
            results.append(dict(bucket=APP_BUCKET,
                                filename=file.filename,
                                classifications=classifications))
    
    data = dict(bucket_name=APP_BUCKET.name, 
                min_confidence=min_confidence, 
                results=results)
    return flask.render_template('image_classify.html', data=data)


@app.route('/image_classify_cloud_vision', methods=['POST', 'GET'])
def image_classify_cloud_vision():
    files = flask.request.files.getlist('files')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"imposing-kite-140412-c3719b8fafdc.json"
    results = []
    if len(files) > 1 or files[0].filename != '':
        for file in files:

            blob = storage.Blob(file.filename, APP_BUCKET)
            blob.upload_from_file(file, blob, content_type=file.mimetype)
            blob.make_public()

            client = vision.ImageAnnotatorClient()

            file.seek(0)

            image = vision.Image(content=file.read())
            response = client.label_detection(image=image)
            labels = response.label_annotations

            label = []
            for l in labels:
                label.append({'description': l.description, 'score': round(l.score, 2)})
            
            logging.info('image_classify: filename={} blob={} classifications={}'\
                .format(file.filename,blob.name,label))
            results.append(dict(bucket=APP_BUCKET,
                                filename=file.filename,
                                classifications=label))
    
    data = dict(bucket_name=APP_BUCKET.name,
                results=results)
    return flask.render_template('image_classify_cloud_vision.html', data=data)


if __name__ == '__main__':
    # When invoked as a program.
    logging.info('Starting app')
    #app.run(host='127.0.0.1', port=8080, debug=True)
    #app.run(host='0.0.0.0', port=443, debug=True)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
