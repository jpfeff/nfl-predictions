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
  const [testSize, setTestSize] = useState(20)

  const [loading, setLoading] = useState(false);

  const [messageApi, contextHolder] = message.useMessage();

  const warning = () => {
    messageApi.open({
      type: 'warning',
      content: 'Please select at least one feature',
    });
  };


  const handlePredict = async () => {
    setLoading(true);
    if (fields.length === 0) {
      setLoading(false);
      warning()
      return
    }
    try {
      const response = await apiRequest({
        method: 'post',
        url: 'https://cs89-project-backend.onrender.com/accuracy',
        data: {
          binary: binary,
          fields: fields,
          model: model,
          start_year: startYear,
          end_year: endYear,
          test_size: testSize * 0.01
        },
      });
  
      setPrediction(response.data);
    } catch (error) {
      console.error('Error predicting:', error);
    } finally {
      setLoading(false); 
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
            <div className="data">
              <div className='slider' style={{marginBottom: '-20px'}}>
                <Slider 
                    min={2013} 
                    max={2022} 
                    range 
                    defaultValue={[2013, 2022]}
                    marks={{2013: '2013', 2022: '2022'}}
                    onChange={(value) => {
                      setStartYear(value[0])
                      setEndYear(value[1])
                    }}
                    value={[startYear, endYear]}
                ></Slider>
              </div>

            </div>
          </div>
          <div className="section">
            <Divider orientation="center">4. Choose test set size</Divider>
            <div className="data">
              <div className='slider'>
                <Slider
                  min={10}
                  max={50}
                  step={10}
                  defaultValue={0}
                  marks={{10: '10%', 20: '20%', 30: '30%', 40: '40%', 50: '50%'}}
                  value={testSize}
                  onChange={(value) => setTestSize(value)}
                ></Slider>
                </div>
              </div>
          </div>
          <div className="section">
            <Divider orientation="center">5. Choose training data features</Divider>
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
            <Divider orientation="center">6. Train model and evaluate accuracy</Divider>
            <Button type="primary" loading={loading} onClick={handlePredict}>Compute Model Accuracy</Button>
          </div>
          {prediction !== null &&
            <Progress type="circle" percent={(prediction * 100).toFixed(2)} />
          }
         
      </div>
  )

}

export default Predictor