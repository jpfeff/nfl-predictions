import axios from 'axios';

const apiRequest = async (config) => {
  const response = await axios({
    ...config,
  });

  return response;
};

export default apiRequest;
