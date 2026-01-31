module.exports = {
  run: [
    {
      method: "fs.link",
      params: {
        venv: "app/env"
      }
    },
    {
      method: "fs.link",
      params: {
        drive: {
          "models": "app/models"
        }
      }
    }
  ]
}
