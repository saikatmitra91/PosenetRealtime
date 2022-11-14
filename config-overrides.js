module.exports = function override(config, env) {
    if (process.env.NODE_ENV === 'production')
    {
      config.resolve.mainFields = ['main'];
    }
    return config;
  }