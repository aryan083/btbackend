const HttpError = require("./httpError");
const ClientError = require("./clientError");
const ServerError = require("./serverError");


module.exports = {
	HttpError,
	...ClientError,
	...ServerError,
};
