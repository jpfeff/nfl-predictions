import { useState } from "react"
import apiRequest from "../services"
import { Radio, Button, Select, Slider, Divider, Progress, message} from 'antd';
import { binaryModels, regressionModels, modelAbbreviations, statAbbreviations} from "../info/constants";
import './index.css';

function Predictor() {
  const [binary, setBinary] = useState(true)
  const [startYear, setStartYear] = useState(2013)
  const [endYear, setEndYear] = useState(2022)
  const [fields, setFields] = useState([])
  const [model, setModel] = useState('LR')
  const [prediction, setPrediction] = useState(0)

  const [loading, setLoading] = useState(false);

  const [messageApi, contextHolder] = message.useMessage();

  const warning = () => {
    messageApi.open({
      type: 'warning',
      content: 'Please select at least one feature',
    });
  };


  const handlePredict = async () => {
    if (fields.length === 0) {
      warning()
      return
    }
    try {
      setLoading(true); // Set loading to true before making the API call
  
      const response = await apiRequest({
        method: 'post',
        url: 'http://127.0.0.1:8000/accuracy',
        data: {
          binary: binary,
          fields: fields,
          model: model,
          start_year: startYear,
          end_year: endYear,
        },
      });
  
      setPrediction(response.data);
      console.log('Predicting...');
      console.log(binary, fields, model, startYear, endYear);
    } catch (error) {
      console.error('Error predicting:', error);
    } finally {
      setLoading(false); // Set loading back to false after the API call completes
    }
  };
  

  const handleBinaryChange = (e) => {
    // if the model is not in the list of binary models, change the model to the first binary model
    // the abbreviations map from the model to its abbreviation, and the value is the abbreviation

    // switching to binary
    if (e.target.value) {
      if (!binaryModels.includes(model)) {
        setModel(binaryModels[0])
      }
    } else {
      if (!regressionModels.includes(model)) {
        setModel(regressionModels[0])
      }
    }
    setBinary(e.target.value)
  }

  return (
      <div className="predict-wrapper">
        {contextHolder}
          <div className="section">
            <Divider orientation="center">1. Choose encoding scheme</Divider>
            <Radio.Group onChange={(e) => handleBinaryChange(e)} value={binary}>
              <Radio.Button value={true}>Binary Encoding</Radio.Button>
              <Radio.Button value={false}>Non-Binary Encoding</Radio.Button>
            </Radio.Group>
          </div>
          <div className="section">
            <Divider orientation="center">2. Choose model</Divider>
            <Radio.Group onChange={(e) => setModel(e.target.value)} value={model}>
              {binary ? 
              binaryModels.map((model) => <Radio.Button key={model} value={model}>{modelAbbreviations[model]}</Radio.Button>) : regressionModels.map((model) => <Radio.Button key={model} value={model}>{modelAbbreviations[model]}</Radio.Button>)}
            </Radio.Group>
          </div>
          <div>
            <Divider orientation="center">3. Choose training data years</Divider>
            <Slider min={2013} 
                    max={2022} 
                    range 
                    defaultValue={[2013, 2022]}
                    marks={{2013: '2013', 2022: '2022'}}
                    onChange={(value) => {
                      setStartYear(value[0])
                      setEndYear(value[1])
                    }}
                    value={[startYear, endYear]}
                     />
          </div>
          <div className="section">
            <Divider orientation="center">4. Choose training data features</Divider>
            <div className="select-all">
              <Select
                placeholder="Select features"
                mode="multiple"
                onChange={(values) => setFields(values)}
                value={fields}
                >
                {Object.keys(statAbbreviations).map((stat) => (
                  <Select.Option key={stat} value={stat}>
                    {statAbbreviations[stat]}
                  </Select.Option>
                ))}
              </Select>
              <Button onClick={() => setFields(Object.keys(statAbbreviations))}>Select All</Button>
              <Button onClick={() => setFields([])}>Clear All</Button>
            </div>
            
          </div>
          <div className="section">
            <Divider orientation="center">5. Train Model and Evaluate Accuracy</Divider>
            <Button type="primary" loading={loading} onClick={handlePredict}>Compute Model Accuracy</Button>
          </div>
          {prediction !== null &&
            <Progress type="circle" percent={(prediction * 100).toFixed(2)} />
          }
         
      </div>
  )

}

export default Predictor